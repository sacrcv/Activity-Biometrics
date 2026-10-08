"""Training loops for ABNet (stage 1) and the GaitGL teacher (stage 0)."""

import json
import os
import time
from typing import Dict, Optional

import torch

from abnet.engine.optim import build_optimizer, build_scheduler
from abnet.utils.dist import is_main_process
from abnet.utils.logger import MetricLogger, get_logger
from abnet.utils.misc import load_checkpoint, save_checkpoint

_AMP_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "none": None, "fp32": None}


def resolve_amp_dtype(name: Optional[str]) -> Optional[torch.dtype]:
    if name is None:
        return None
    key = str(name).lower()
    if key not in _AMP_DTYPES:
        raise ValueError(f"run.amp must be one of {sorted(_AMP_DTYPES)}, got '{name}'")
    return _AMP_DTYPES[key]


class BaseTrainer:
    """Shared training machinery: AMP, accumulation, clipping, checkpointing.

    Subclasses provide :meth:`forward_step` (one batch -> loss dict) and
    :attr:`metric_keys` (which of its entries to log).
    """

    metric_keys = ("loss",)

    def __init__(
        self,
        cfg,
        model,
        criterion,
        train_loader,
        device: torch.device,
        output_dir: str = "output",
    ):
        self.cfg = cfg
        self.model = model
        self.criterion = criterion
        self.train_loader = train_loader
        self.device = device
        self.output_dir = output_dir
        self.logger = get_logger()

        run_cfg = cfg.run
        self.max_epoch = run_cfg.get("max_epoch", 150)
        self.accum_grad_iters = max(run_cfg.get("accum_grad_iters", 1), 1)
        self.clip_grad_norm = run_cfg.get("clip_grad_norm", 5.0)
        self.print_freq = run_cfg.get("print_freq", 20)
        self.save_freq = run_cfg.get("save_freq", 10)

        self.amp_dtype = resolve_amp_dtype(run_cfg.get("amp", "bf16"))
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=(self.amp_dtype == torch.float16 and device.type == "cuda")
        )

        self.optimizer = build_optimizer(cfg, model)
        self.scheduler = build_scheduler(cfg, self.optimizer, len(train_loader))

        self.start_epoch = 0
        self.best_score = -1.0
        self.history = []

    # ------------------------------------------------------------------
    def resume(self, path: str) -> None:
        ckpt = load_checkpoint(
            path, self.model, self.optimizer, self.scheduler, self.scaler, strict=False
        )
        self.start_epoch = int(ckpt.get("epoch", -1)) + 1
        best = ckpt.get("best")
        if isinstance(best, (int, float)):
            self.best_score = float(best)
        self.logger.info(f"resuming at epoch {self.start_epoch} (best={self.best_score:.2f})")

    def _move_batch(self, batch: Dict) -> Dict:
        out = {}
        for key, value in batch.items():
            out[key] = (
                value.to(self.device, non_blocking=True)
                if isinstance(value, torch.Tensor)
                else value
            )
        return out

    def _autocast(self):
        if self.amp_dtype is None or self.device.type != "cuda":
            return torch.autocast(device_type="cpu", enabled=False)
        return torch.autocast(device_type="cuda", dtype=self.amp_dtype)

    def forward_step(self, batch: Dict) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    # ------------------------------------------------------------------
    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        self.criterion.train()
        sampler = getattr(self.train_loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

        metric_logger = MetricLogger()
        header = f"epoch {epoch}/{self.max_epoch - 1}"
        self.optimizer.zero_grad(set_to_none=True)

        for step, batch in metric_logger.log_every(
            self.train_loader, self.print_freq, header=header
        ):
            batch = self._move_batch(batch)

            with self._autocast():
                losses = self.forward_step(batch)
                loss = losses["loss"] / self.accum_grad_iters

            if not torch.isfinite(loss):
                self.logger.warning(f"non-finite loss at epoch {epoch} step {step}; skipping")
                self.optimizer.zero_grad(set_to_none=True)
                continue

            self.scaler.scale(loss).backward()

            if (step + 1) % self.accum_grad_iters == 0:
                if self.clip_grad_norm and self.clip_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        self.clip_grad_norm,
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                if self.scheduler.granularity == "iter":
                    self.scheduler.step()

            metric_logger.update(
                lr=self.scheduler.get_last_lr()[0],
                **{k: losses[k] for k in self.metric_keys if k in losses},
            )

        if self.scheduler.granularity == "epoch":
            self.scheduler.step()

        stats = metric_logger.global_avg_dict()
        self.logger.info(
            f"{header} summary: " + "  ".join(f"{k}={v:.4f}" for k, v in stats.items())
        )
        return stats

    # ------------------------------------------------------------------
    def can_evaluate(self) -> bool:
        return False

    def evaluate(self) -> Dict:
        raise NotImplementedError

    def _score(self, results: Dict) -> float:
        raise NotImplementedError

    def _dump_history(self) -> None:
        if not is_main_process():
            return
        with open(os.path.join(self.output_dir, "history.json"), "w") as f:
            json.dump(self.history, f, indent=2)

    def run(self) -> Dict:
        start = time.time()
        for epoch in range(self.start_epoch, self.max_epoch):
            stats = self.train_one_epoch(epoch)
            entry = {"epoch": epoch, "train": stats}

            is_last = epoch == self.max_epoch - 1
            eval_freq = self.cfg.run.get("eval_freq", 10)
            should_eval = self.can_evaluate() and (
                is_last or (eval_freq > 0 and (epoch + 1) % eval_freq == 0)
            )
            if should_eval:
                results = self.evaluate()
                entry["eval"] = results
                score = self._score(results)
                if score > self.best_score:
                    self.best_score = score
                    save_checkpoint(
                        os.path.join(self.output_dir, "checkpoint_best.pth"),
                        self.model, self.optimizer, self.scheduler, self.scaler,
                        epoch=epoch, best=self.best_score, cfg=self.cfg,
                    )
                    self.logger.info(f"new best score = {self.best_score:.2f}")

            if is_last or (self.save_freq > 0 and (epoch + 1) % self.save_freq == 0):
                save_checkpoint(
                    os.path.join(self.output_dir, "checkpoint_last.pth"),
                    self.model, self.optimizer, self.scheduler, self.scaler,
                    epoch=epoch, best=self.best_score, cfg=self.cfg,
                )

            self.history.append(entry)
            self._dump_history()

        elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start))
        self.logger.info(f"training finished in {elapsed}; best={self.best_score:.2f}")
        return {"best": self.best_score, "history": self.history}


class Trainer(BaseTrainer):
    """Stage 1: joint biometrics and activity training of ABNet."""

    metric_keys = (
        "loss", "loss_bio", "loss_ce", "loss_tri",
        "loss_ac", "loss_kd", "loss_dis", "id_acc", "act_acc",
    )

    def __init__(
        self,
        cfg,
        model,
        criterion,
        train_loader,
        device: torch.device,
        query_loader=None,
        gallery_loader=None,
        output_dir: str = "output",
    ):
        super().__init__(cfg, model, criterion, train_loader, device, output_dir)
        self.query_loader = query_loader
        self.gallery_loader = gallery_loader

        run_cfg = cfg.run
        self.protocols = list(run_cfg.get("protocols", ["same_activity"]))
        self.primary_protocol = run_cfg.get("primary_protocol", self.protocols[0])
        self.primary_metric = run_cfg.get("primary_metric", "R@1")

    def forward_step(self, batch: Dict) -> Dict[str, torch.Tensor]:
        outputs = self.model(batch["frames"], batch.get("frames_distorted"))
        return self.criterion(outputs, batch)

    def can_evaluate(self) -> bool:
        return self.query_loader is not None and self.gallery_loader is not None

    def evaluate(self) -> Dict[str, Dict[str, float]]:
        from abnet.evaluation import evaluate

        run_cfg = self.cfg.run
        return evaluate(
            self.model,
            self.query_loader,
            self.gallery_loader,
            self.device,
            protocols=self.protocols,
            amp_dtype=self.amp_dtype,
            max_rank=run_cfg.get("max_rank", 50),
            print_freq=self.print_freq,
            feature=run_cfg.get("eval_feature", "fused"),
            far_targets=run_cfg.get("far_targets", [0.001]),
        )

    def _score(self, results: Dict[str, Dict[str, float]]) -> float:
        block = results.get(self.primary_protocol) or next(iter(results.values()))
        return float(block.get(self.primary_metric, block.get("R@1", 0.0)))


class TeacherTrainer(BaseTrainer):
    """Stage 0: train GaitGL on silhouettes to become the bias-less teacher.

    There is no retrieval evaluation here. The teacher's only job is to supply
    a calibrated identity distribution over the *train* identities, so train
    accuracy is the quantity that matters and the last checkpoint is the one
    stage 1 consumes.
    """

    metric_keys = ("loss", "loss_ce", "loss_tri", "id_acc")

    def forward_step(self, batch: Dict) -> Dict[str, torch.Tensor]:
        if "silhouette" not in batch:
            raise KeyError(
                "the teacher needs silhouettes, but the batch has none. Run "
                "tools/extract_silhouettes.py and rebuild the annotation with "
                "tools/prepare_dataset.py so samples carry 'silhouettes_dir'."
            )
        outputs = self.model(batch["silhouette"])
        return self.criterion(outputs, batch)

    def _score(self, results: Dict) -> float:
        return 0.0
