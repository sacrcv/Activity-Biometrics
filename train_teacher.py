#!/usr/bin/env python
"""Stage 0: train the bias-less teacher T (GaitGL on silhouettes).

The teacher must exist before ABNet can be trained, because Eq. 1 distils its
identity distribution. It is "bias-less" because its only input is a binary
silhouette, so it cannot have learned clothing colour, background or any other
appearance cue -- which is exactly what makes it a useful target for pushing
appearance out of f_bb.

Prerequisites: silhouettes extracted with ``tools/extract_silhouettes.py`` and
an annotation file whose samples carry ``silhouettes_dir``.

Usage::

    # single GPU
    python train_teacher.py --cfg configs/teacher_ntu_rgb_ab.yaml

    # multi-GPU
    torchrun --nproc_per_node=4 train_teacher.py --cfg configs/teacher_ntu_rgb_ab.yaml

    # wrapper (picks the GPU count up from CUDA_VISIBLE_DEVICES)
    bash scripts/train_teacher.sh configs/teacher_ntu_rgb_ab.yaml

    # inline config overrides
    python train_teacher.py --cfg configs/teacher_ntu_rgb_ab.yaml \\
        run.max_epoch=30 run.lr=5e-5

The resulting ``checkpoint_last.pth`` is what ``teacher.checkpoint`` in the
stage-1 config should point at. Note that the teacher is tied to one dataset's
identity label space, so each benchmark needs its own.
"""

import argparse
import os
import sys

import torch

from abnet.data import build_datasets, build_train_loader
from abnet.engine import TeacherCriterion, TeacherTrainer
from abnet.models import build_gaitgl
from abnet.utils import (
    config_to_str,
    count_parameters,
    destroy_distributed,
    get_logger,
    init_distributed_mode,
    is_main_process,
    load_config,
    save_config,
    set_seed,
    setup_logger,
)


def main() -> int:
    args = parse_args()
    cfg = load_config(args.cfg, args.overrides)

    distributed, rank, world_size, local_rank = init_distributed_mode()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    output_dir = args.output or cfg.run.get("output_dir") or "output/teacher"
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_dir)
    if is_main_process():
        os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger(os.path.join(output_dir, "train_teacher.log") if is_main_process() else None)
    logger.info(f"output dir: {output_dir}")
    logger.info(f"world size {world_size}, device {device}")
    logger.info("config:\n" + config_to_str(cfg))
    if is_main_process():
        save_config(cfg, os.path.join(output_dir, "config.yaml"))

    set_seed(cfg.run.get("seed", 42), rank=rank, deterministic=cfg.run.get("deterministic", False))

    datasets, meta = build_datasets(cfg, for_teacher=True)
    if "train" not in datasets:
        logger.error("the annotation has no train split; nothing to train the teacher on")
        return 1

    train_dataset = datasets["train"]
    with_silhouettes = sum(1 for s in train_dataset.samples if s.get("silhouettes_dir"))
    if with_silhouettes == 0:
        logger.error(
            "no train clip has a 'silhouettes_dir'. Run tools/extract_silhouettes.py, "
            "then rebuild the annotation with tools/prepare_dataset.py "
            "--silhouettes-root <dir>."
        )
        return 1
    logger.info(
        f"{with_silhouettes}/{len(train_dataset)} train clips have silhouettes"
    )

    model = build_gaitgl(
        num_classes=meta["num_identities"], cfg=cfg.get("teacher", {})
    ).to(device)
    params = count_parameters(model)
    logger.info(
        f"GaitGL parameters: total={params['total'] / 1e6:.2f}M, "
        f"trainable={params['trainable'] / 1e6:.2f}M"
    )

    criterion = TeacherCriterion(
        num_identities=meta["num_identities"],
        margin=cfg.get("loss", {}).get("margin", 0.3),
        label_smoothing=cfg.get("loss", {}).get("label_smoothing", 0.0),
    ).to(device)

    train_loader = build_train_loader(cfg, train_dataset)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            find_unused_parameters=cfg.run.get("find_unused_parameters", False),
        )

    trainer = TeacherTrainer(
        cfg=cfg,
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        device=device,
        output_dir=output_dir,
    )
    resume = args.resume or cfg.run.get("resume")
    if resume:
        trainer.resume(resume)

    trainer.run()
    logger.info(
        "teacher ready. Point the stage-1 config at it:\n"
        f"  teacher.checkpoint={os.path.join(output_dir, 'checkpoint_last.pth')}"
    )
    destroy_distributed()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--cfg", required=True, help="path to a teacher_*.yaml config")
    parser.add_argument("--output", default=None, help="override run.output_dir")
    parser.add_argument("--resume", default=None, help="checkpoint to resume from")
    parser.add_argument("overrides", nargs="*", help="config overrides, e.g. run.lr=1e-4")
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main())
