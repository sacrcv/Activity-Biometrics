#!/usr/bin/env python
"""Stage 1: train ABNet.

Runs the full objective of Eq. 6 -- biometrics loss, joint activity learning,
bias-less distillation from the frozen teacher, and bias learning through
biometrics distortion.

Prerequisites: an annotation file (``tools/prepare_dataset.py``) and a trained
teacher (``train_teacher.py``). Training without a teacher is possible with
``teacher.enabled=false``, but that is the paper's "w/o K/D" ablation rather
than the method.

Usage::

    # single GPU
    python train.py --cfg configs/ntu_rgb_ab.yaml

    # multi-GPU
    torchrun --nproc_per_node=8 train.py --cfg configs/ntu_rgb_ab.yaml

    # wrapper (picks the GPU count up from CUDA_VISIBLE_DEVICES)
    bash scripts/train.sh configs/ntu_rgb_ab.yaml

    # inline config overrides
    python train.py --cfg configs/ntu_rgb_ab.yaml run.batch_size=16 run.max_epoch=30

    # resume
    python train.py --cfg configs/ntu_rgb_ab.yaml --resume output/ntu_rgb_ab/checkpoint_last.pth
"""

import argparse
import os
import sys

import torch

from abnet.data import build_datasets, build_eval_loader, build_train_loader
from abnet.engine import ABNetCriterion, Trainer
from abnet.models import ABNetModel, build_gaitgl
from abnet.utils import (
    config_to_str,
    count_parameters,
    destroy_distributed,
    init_distributed_mode,
    is_main_process,
    load_config,
    save_config,
    set_seed,
    setup_logger,
)


def load_teacher(cfg, num_identities: int, device, logger):
    """Build the frozen GaitGL teacher, or None when distillation is off."""
    teacher_cfg = cfg.get("teacher", {}) or {}
    if not teacher_cfg.get("enabled", True):
        logger.warning("teacher.enabled=false: running the 'w/o K/D' ablation, not ABNet")
        return None

    checkpoint = teacher_cfg.get("checkpoint")
    if not checkpoint:
        logger.error(
            "teacher.enabled is true but teacher.checkpoint is unset. Train the "
            "teacher first:\n  bash scripts/train_teacher.sh configs/teacher_<dataset>.yaml\n"
            "or set teacher.enabled=false to run without distillation."
        )
        raise SystemExit(2)

    path = checkpoint
    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    if not os.path.exists(path):
        logger.error(f"teacher checkpoint not found: {path}")
        raise SystemExit(2)

    teacher = build_gaitgl(num_classes=num_identities, cfg=teacher_cfg)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = payload.get("model", payload)
    result = teacher.load_state_dict(state_dict, strict=False)
    if result.missing_keys:
        logger.warning(
            f"teacher checkpoint is missing {len(result.missing_keys)} tensor(s), "
            f"e.g. {result.missing_keys[:4]}"
        )
    if result.unexpected_keys:
        logger.warning(
            f"teacher checkpoint has {len(result.unexpected_keys)} unexpected tensor(s), "
            f"e.g. {result.unexpected_keys[:4]}"
        )
    logger.info(f"loaded teacher from {path} (epoch {payload.get('epoch')})")
    return teacher.to(device)


def main() -> int:
    args = parse_args()
    cfg = load_config(args.cfg, args.overrides)

    distributed, rank, world_size, local_rank = init_distributed_mode()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    output_dir = args.output or cfg.run.get("output_dir") or "output/abnet"
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_dir)
    if is_main_process():
        os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger(os.path.join(output_dir, "train.log") if is_main_process() else None)
    logger.info(f"output dir: {output_dir}")
    logger.info(f"world size {world_size}, device {device}")
    logger.info("config:\n" + config_to_str(cfg))
    if is_main_process():
        save_config(cfg, os.path.join(output_dir, "config.yaml"))

    set_seed(cfg.run.get("seed", 42), rank=rank, deterministic=cfg.run.get("deterministic", False))

    datasets, meta = build_datasets(cfg)
    if "train" not in datasets:
        logger.error("the annotation has no train split; nothing to train on")
        return 1

    model = ABNetModel.from_config(
        cfg, num_identities=meta["num_identities"], num_actions=meta["num_actions"]
    ).to(device)
    params = count_parameters(model)
    logger.info(
        f"ABNet parameters: total={params['total'] / 1e6:.2f}M, "
        f"trainable={params['trainable'] / 1e6:.2f}M, frozen={params['frozen'] / 1e6:.2f}M"
    )

    teacher = load_teacher(cfg, meta["num_identities"], device, logger)
    criterion = ABNetCriterion.from_config(
        cfg,
        num_identities=meta["num_identities"],
        num_actions=meta["num_actions"],
        multilabel=meta["multilabel"],
        teacher=teacher,
    ).to(device)

    train_loader = build_train_loader(cfg, datasets["train"])
    query_loader = build_eval_loader(cfg, datasets["query"]) if "query" in datasets else None
    gallery_loader = (
        build_eval_loader(cfg, datasets["gallery"]) if "gallery" in datasets else None
    )

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            find_unused_parameters=cfg.run.get("find_unused_parameters", False),
        )

    trainer = Trainer(
        cfg=cfg,
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        device=device,
        query_loader=query_loader,
        gallery_loader=gallery_loader,
        output_dir=output_dir,
    )
    resume = args.resume or cfg.run.get("resume")
    if resume:
        trainer.resume(resume)

    trainer.run()
    destroy_distributed()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--cfg", required=True, help="path to a dataset config")
    parser.add_argument("--output", default=None, help="override run.output_dir")
    parser.add_argument("--resume", default=None, help="checkpoint to resume from")
    parser.add_argument(
        "overrides", nargs="*", help="config overrides, e.g. run.batch_size=16"
    )
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main())
