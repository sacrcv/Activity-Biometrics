"""Distributed helpers and the LAVIS checkpoint cache utility.

``download_cached_file`` is taken from LAVIS (``lavis/common/dist_utils.py``,
BSD-3-Clause) because ``eva_vit.create_eva_vit_g`` depends on it to fetch
``eva_vit_g.pth`` into the timm hub cache.
"""

import datetime
import functools
import os

import torch
import torch.distributed as dist


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def get_rank() -> int:
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def is_main_process() -> bool:
    return get_rank() == 0


def barrier() -> None:
    if is_dist_avail_and_initialized():
        dist.barrier()


def main_process(func):
    """Decorator: only execute ``func`` on rank 0."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if is_main_process():
            return func(*args, **kwargs)
        return None

    return wrapper


def init_distributed_mode():
    """Initialise ``torch.distributed`` from torchrun environment variables.

    Returns ``(distributed, rank, world_size, local_rank)``. Falls back to
    single-process mode when the torchrun variables are absent.
    """
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        return False, 0, 1, 0

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank % max(torch.cuda.device_count(), 1)))

    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl" if torch.cuda.is_available() else "gloo",
        init_method="env://",
        world_size=world_size,
        rank=rank,
        timeout=datetime.timedelta(hours=2),
    )
    dist.barrier()
    return True, rank, world_size, local_rank


def destroy_distributed():
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()


def all_gather_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Concatenate ``tensor`` from every rank along dim 0 (no gradient flow)."""
    world_size = get_world_size()
    if world_size == 1:
        return tensor
    gathered = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor.contiguous())
    return torch.cat(gathered, dim=0)


def get_cache_dir() -> str:
    """Torch hub checkpoint cache, honouring ``TORCH_HOME``.

    Same location timm used (``~/.cache/torch/hub/checkpoints``), so weights
    already downloaded by other BLIP-2 codebases are reused rather than
    re-fetched.
    """
    cache_dir = os.path.join(torch.hub.get_dir(), "checkpoints")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def download_cached_file(url, check_hash=True, progress=True):
    """Download ``url`` into the torch hub cache once, on rank 0 only.

    Follows LAVIS ``lavis/common/dist_utils.py`` in behaviour; the timm
    dependency it used has been replaced with ``torch.hub``.
    """
    filename = os.path.basename(torch.hub.urlparse(url).path)
    cached_file = os.path.join(get_cache_dir(), filename)

    if is_main_process() and not os.path.exists(cached_file):
        hash_prefix = None
        if check_hash:
            # torch's convention: "name-<sha256 prefix>.pth".
            match = torch.hub.HASH_REGEX.search(filename)
            hash_prefix = match.group(1) if match else None
        torch.hub.download_url_to_file(
            url, cached_file, hash_prefix=hash_prefix, progress=progress
        )

    # Other ranks wait for rank 0 to finish writing the file.
    barrier()

    return cached_file


def is_url(url_or_filename: str) -> bool:
    from urllib.parse import urlparse

    parsed = urlparse(str(url_or_filename))
    return parsed.scheme in ("http", "https")
