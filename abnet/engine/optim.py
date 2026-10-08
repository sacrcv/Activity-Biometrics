"""Optimiser and learning-rate schedule.

The paper: "Adam [25] is used as the optimizer with weight decay of 5x10^-4 and
learning rate of 3.5x10^-4. The model is trained for 150 epochs with a decay
factor 0.1 after every 40 epochs."

So the default here is plain Adam with a step decay, stepped **per epoch**
(a factor-of-10 drop every 40 epochs only makes sense on an epoch clock).
``WarmupCosineSchedule`` is kept as an opt-in alternative via
``run.scheduler: cosine``, stepped per iteration; it is not what the paper
used, but it is the more forgiving choice when fine-tuning the transformer
heads at a short epoch budget.
"""

import math
from typing import List

import torch

from abnet.utils.logger import get_logger

#: Submodules that are randomly initialised rather than Kinetics-pretrained.
#: Giving them a larger LR via ``run.head_lr_mult`` is the usual re-ID recipe.
HEAD_PREFIXES = ("actor_head.", "activity_head.", "identity_head.")


def build_optimizer(cfg, model: torch.nn.Module) -> torch.optim.Optimizer:
    """Adam (or AdamW) with weight decay disabled for biases, norms and queries."""
    run_cfg = cfg.run
    weight_decay = run_cfg.get("weight_decay", 5e-4)
    base_lr = run_cfg.get("lr", 3.5e-4)
    betas = tuple(run_cfg.get("betas", [0.9, 0.999]))
    head_lr_mult = run_cfg.get("head_lr_mult", 1.0)
    kind = str(run_cfg.get("optimizer", "adam")).lower()

    groups = {
        "decay": {"params": [], "weight_decay": weight_decay, "lr": base_lr},
        "no_decay": {"params": [], "weight_decay": 0.0, "lr": base_lr},
        "head_decay": {"params": [], "weight_decay": weight_decay, "lr": base_lr * head_lr_mult},
        "head_no_decay": {"params": [], "weight_decay": 0.0, "lr": base_lr * head_lr_mult},
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_head = name.startswith(HEAD_PREFIXES)
        # 1-D tensors are norms/biases; the decoder's query table is
        # embedding-like and conventionally excluded from decay too.
        no_decay = param.ndim <= 1 or name.endswith(".bias") or name.endswith(".query")
        key = ("head_" if is_head else "") + ("no_decay" if no_decay else "decay")
        groups[key]["params"].append(param)

    param_groups = [g for g in groups.values() if g["params"]]
    if not param_groups:
        raise ValueError("no trainable parameters found")

    get_logger().info(
        f"optimizer '{kind}' groups: "
        + ", ".join(
            f"{name}={len(g['params'])} tensors (lr={g['lr']:.2e}, wd={g['weight_decay']})"
            for name, g in groups.items()
            if g["params"]
        )
    )

    if kind == "adam":
        return torch.optim.Adam(param_groups, lr=base_lr, betas=betas)
    if kind == "adamw":
        return torch.optim.AdamW(param_groups, lr=base_lr, betas=betas)
    if kind == "sgd":
        return torch.optim.SGD(
            param_groups, lr=base_lr, momentum=run_cfg.get("momentum", 0.9), nesterov=True
        )
    raise ValueError(f"run.optimizer must be 'adam', 'adamw' or 'sgd', got '{kind}'")


class _BaseSchedule:
    """Shared LR-ratio plumbing; subclasses define :meth:`_ratio`."""

    #: "epoch" schedules are stepped once per epoch, "iter" once per step.
    granularity = "iter"

    def __init__(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer
        self.base_lrs: List[float] = [g["lr"] for g in optimizer.param_groups]
        self.step_count = 0
        self._apply()

    def _ratio(self) -> float:
        raise NotImplementedError

    def _apply(self) -> None:
        ratio = self._ratio()
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * ratio

    def step(self) -> None:
        self.step_count += 1
        self._apply()

    def get_last_lr(self) -> List[float]:
        return [g["lr"] for g in self.optimizer.param_groups]

    def state_dict(self):
        return {"step_count": self.step_count, "base_lrs": self.base_lrs}

    def load_state_dict(self, state):
        self.step_count = state["step_count"]
        self.base_lrs = state["base_lrs"]
        self._apply()


class StepDecaySchedule(_BaseSchedule):
    """The paper's schedule: multiply the LR by ``gamma`` every ``step_size``
    epochs, with an optional linear warmup measured in epochs.

    Args:
        step_size: epochs between decays (40).
        gamma: decay factor (0.1).
        warmup_epochs: 0 reproduces the paper exactly.
        warmup_start_ratio: LR multiplier at the very first epoch of warmup.
    """

    granularity = "epoch"

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        step_size: int = 40,
        gamma: float = 0.1,
        warmup_epochs: int = 0,
        warmup_start_ratio: float = 0.01,
    ):
        self.step_size = max(int(step_size), 1)
        self.gamma = float(gamma)
        self.warmup_epochs = max(int(warmup_epochs), 0)
        self.warmup_start_ratio = float(warmup_start_ratio)
        super().__init__(optimizer)

    def _ratio(self) -> float:
        epoch = self.step_count
        if epoch < self.warmup_epochs:
            progress = epoch / max(self.warmup_epochs, 1)
            return self.warmup_start_ratio + (1.0 - self.warmup_start_ratio) * progress
        effective = epoch - self.warmup_epochs
        return self.gamma ** (effective // self.step_size)


class WarmupCosineSchedule(_BaseSchedule):
    """Linear warmup then cosine decay, stepped once per iteration.

    Each parameter group decays from its own initial LR, so a ``head_lr_mult``
    ratio is preserved throughout training.
    """

    granularity = "iter"

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        warmup_steps: int = 0,
        min_lr_ratio: float = 0.01,
        warmup_start_ratio: float = 0.01,
    ):
        self.total_steps = max(total_steps, 1)
        self.warmup_steps = max(min(warmup_steps, self.total_steps - 1), 0)
        self.min_lr_ratio = min_lr_ratio
        self.warmup_start_ratio = warmup_start_ratio
        super().__init__(optimizer)

    def _ratio(self) -> float:
        step = self.step_count
        if step < self.warmup_steps:
            progress = step / max(self.warmup_steps, 1)
            return self.warmup_start_ratio + (1.0 - self.warmup_start_ratio) * progress
        progress = (step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine


def build_scheduler(cfg, optimizer, steps_per_epoch: int):
    """Build the schedule named by ``run.scheduler`` (``step`` or ``cosine``)."""
    run_cfg = cfg.run
    kind = str(run_cfg.get("scheduler", "step")).lower()
    max_epoch = run_cfg.get("max_epoch", 150)

    if kind == "step":
        return StepDecaySchedule(
            optimizer,
            step_size=run_cfg.get("lr_decay_epochs", 40),
            gamma=run_cfg.get("lr_decay_gamma", 0.1),
            warmup_epochs=run_cfg.get("warmup_epochs", 0),
            warmup_start_ratio=run_cfg.get("warmup_start_ratio", 0.01),
        )
    if kind == "cosine":
        accum = max(run_cfg.get("accum_grad_iters", 1), 1)
        steps = max(steps_per_epoch // accum, 1)
        return WarmupCosineSchedule(
            optimizer,
            total_steps=max_epoch * steps,
            warmup_steps=int(run_cfg.get("warmup_epochs", 1) * steps),
            min_lr_ratio=run_cfg.get("min_lr_ratio", 0.01),
            warmup_start_ratio=run_cfg.get("warmup_start_ratio", 0.01),
        )
    raise ValueError(f"run.scheduler must be 'step' or 'cosine', got '{kind}'")
