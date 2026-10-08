"""Logging setup and running-average meters."""

import logging
import sys
import time
from collections import deque

import torch

from abnet.utils.dist import get_world_size, is_dist_avail_and_initialized, is_main_process

_LOGGER_NAME = "abnet"


def setup_logger(output_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """Create the ``abnet`` logger. Non-main ranks are silenced to WARNING."""
    logger = logging.getLogger(_LOGGER_NAME)
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(level if is_main_process() else logging.WARNING)

    fmt = logging.Formatter(
        fmt="[%(asctime)s %(levelname)s] %(message)s", datefmt="%m/%d %H:%M:%S"
    )

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    if output_file is not None and is_main_process():
        file_handler = logging.FileHandler(output_file, mode="a")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger(_LOGGER_NAME)


class SmoothedValue:
    """Tracks a series of values with a windowed median/average."""

    def __init__(self, window_size: int = 50, fmt: str = "{median:.4f} ({global_avg:.4f})"):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n: int = 1):
        if isinstance(value, torch.Tensor):
            value = value.detach().item()
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        torch.distributed.barrier()
        torch.distributed.all_reduce(t)
        self.count = int(t[0].item())
        self.total = t[1].item()

    @property
    def median(self):
        if not self.deque:
            return 0.0
        return torch.tensor(list(self.deque)).median().item()

    @property
    def avg(self):
        if not self.deque:
            return 0.0
        return torch.tensor(list(self.deque), dtype=torch.float32).mean().item()

    @property
    def global_avg(self):
        return self.total / self.count if self.count else 0.0

    def __str__(self):
        return self.fmt.format(median=self.median, avg=self.avg, global_avg=self.global_avg)


class MetricLogger:
    """Aggregates named :class:`SmoothedValue` meters and prints progress."""

    #: Per-meter format overrides for quantities the default 4-decimal format
    #: would flatten to 0.0000.
    DEFAULT_FORMATS = {"lr": "{global_avg:.2e}"}

    def __init__(self, delimiter: str = "  ", window_size: int = 50, formats=None):
        self.delimiter = delimiter
        self.window_size = window_size
        self.formats = dict(self.DEFAULT_FORMATS)
        if formats:
            self.formats.update(formats)
        self.meters = {}

    def _meter(self, name: str) -> SmoothedValue:
        if name not in self.meters:
            fmt = self.formats.get(name, "{median:.4f} ({global_avg:.4f})")
            self.meters[name] = SmoothedValue(window_size=self.window_size, fmt=fmt)
        return self.meters[name]

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            self._meter(k).update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        raise AttributeError(f"{type(self).__name__} has no attribute {attr}")

    def __str__(self):
        return self.delimiter.join(f"{name}: {meter}" for name, meter in self.meters.items())

    def global_avg_dict(self):
        return {name: meter.global_avg for name, meter in self.meters.items()}

    def log_every(self, iterable, print_freq: int, header: str = ""):
        logger = get_logger()
        start = time.time()
        iter_time = SmoothedValue(fmt="{global_avg:.3f}")
        total = len(iterable) if hasattr(iterable, "__len__") else None
        end = time.time()

        for i, obj in enumerate(iterable):
            yield i, obj
            iter_time.update(time.time() - end)
            end = time.time()

            if print_freq > 0 and (i % print_freq == 0 or (total is not None and i == total - 1)):
                position = f"[{i}/{total}]" if total is not None else f"[{i}]"
                mem = ""
                if torch.cuda.is_available():
                    mem = f"  mem: {torch.cuda.max_memory_allocated() / 2 ** 30:.2f}G"
                eta = ""
                if total is not None and iter_time.global_avg > 0:
                    remaining = int(iter_time.global_avg * (total - i))
                    eta = f"  eta: {time.strftime('%H:%M:%S', time.gmtime(remaining))}"
                logger.info(
                    f"{header} {position}  {self}  t/it: {iter_time}s{eta}{mem}"
                )

        elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start))
        logger.info(f"{header} finished in {elapsed} (world_size={get_world_size()})")
