#!/usr/bin/env python
"""Evaluate an ABNet checkpoint.

Reports rank-1, rank-5, rank-10, rank-20, mAP and TAR @ 0.1% FAR for every
requested protocol. Inference uses the RGB branch only -- no silhouettes, no
teacher, no distortion branch.

Usage::

    bash scripts/test.sh configs/ntu_rgb_ab.yaml output/ntu_rgb_ab/checkpoint_best.pth

    python test.py --cfg configs/ntu_rgb_ab.yaml \\
        --checkpoint output/ntu_rgb_ab/checkpoint_best.pth

    # retrieve with f_bb alone, dropping the activity prior
    python test.py --cfg configs/ntu_rgb_ab.yaml --checkpoint <ckpt> --feature biometrics

    # the appearance feature should score badly: that is the disentanglement check
    python test.py --cfg configs/ntu_rgb_ab.yaml --checkpoint <ckpt> --feature appearance

    # a subset of protocols
    python test.py --cfg configs/ntu_rgb_ab.yaml --checkpoint <ckpt> \\
        --protocols same_activity_include_view,cross_activity_exclude_view
"""

import argparse
import json
import os
import sys

import torch

from abnet.data import build_datasets, build_eval_loader
from abnet.engine import resolve_amp_dtype
from abnet.evaluation import evaluate
from abnet.models import FEATURE_CHOICES, ABNetModel
from abnet.utils import (
    config_to_str,
    destroy_distributed,
    init_distributed_mode,
    is_main_process,
    load_checkpoint,
    load_config,
    set_seed,
    setup_logger,
)


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

    logger = setup_logger(os.path.join(output_dir, "test.log") if is_main_process() else None)
    logger.info(f"world size {world_size}, device {device}")
    logger.info("config:\n" + config_to_str(cfg))

    set_seed(cfg.run.get("seed", 42), rank=rank)

    # The teacher is irrelevant at test time, and loading silhouettes would be
    # wasted IO, so switch it off before the datasets are built.
    cfg.teacher.enabled = False
    datasets, meta = build_datasets(cfg)
    if "query" not in datasets or "gallery" not in datasets:
        logger.error(
            "evaluation needs both a query and a gallery split; rebuild the "
            "annotation with tools/prepare_dataset.py"
        )
        return 1

    model = ABNetModel.from_config(
        cfg, num_identities=meta["num_identities"], num_actions=meta["num_actions"]
    ).to(device)

    checkpoint = args.checkpoint or cfg.run.get("checkpoint")
    if checkpoint:
        load_checkpoint(checkpoint, model, strict=False)
    else:
        logger.warning(
            "no --checkpoint given: evaluating the freshly initialised model, which "
            "only sanity-checks the pipeline. The numbers will be near chance."
        )

    protocols = (
        [p.strip() for p in args.protocols.split(",") if p.strip()]
        if args.protocols
        else list(cfg.run.get("protocols", ["same_activity"]))
    )
    far_targets = [float(f) for f in cfg.run.get("far_targets", [0.001])]

    results = evaluate(
        model,
        build_eval_loader(cfg, datasets["query"]),
        build_eval_loader(cfg, datasets["gallery"]),
        device,
        protocols=protocols,
        amp_dtype=resolve_amp_dtype(cfg.run.get("amp", "bf16")),
        max_rank=cfg.run.get("max_rank", 50),
        print_freq=cfg.run.get("print_freq", 50),
        feature=args.feature,
        far_targets=far_targets,
    )

    if is_main_process():
        path = os.path.join(output_dir, f"results_{args.feature}.json")
        with open(path, "w") as f:
            json.dump(
                {
                    "dataset": meta["name"],
                    "protocol": meta["protocol"],
                    "checkpoint": checkpoint,
                    "feature": args.feature,
                    "far_targets": far_targets,
                    "results": results,
                },
                f,
                indent=2,
            )
        logger.info(f"wrote {path}")

    destroy_distributed()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default=None, help="override run.output_dir")
    parser.add_argument(
        "--feature",
        default="fused",
        choices=list(FEATURE_CHOICES),
        help="retrieval feature; 'fused' is the paper's concat(F_Ac, f_bb)",
    )
    parser.add_argument(
        "--protocols", default=None, help="comma-separated subset of the config's protocols"
    )
    parser.add_argument("overrides", nargs="*", help="config overrides")
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main())
