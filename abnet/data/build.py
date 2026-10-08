"""Build datasets and dataloaders from a config."""

import os
from typing import Dict, Optional, Tuple

from torch.utils.data import DataLoader

from abnet.data.dataset import (
    VideoReIDDataset,
    build_pid_map,
    collate_video_batch,
    load_annotation,
    log_dataset_summary,
)
from abnet.data.samplers import (
    DistributedRandomIdentitySampler,
    InferenceSampler,
    RandomIdentitySampler,
    seed_worker,
)
from abnet.data.transforms import build_transform
from abnet.utils.config import repo_root
from abnet.utils.dist import get_world_size, is_dist_avail_and_initialized
from abnet.utils.logger import get_logger


def _abs(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    return path if os.path.isabs(path) else os.path.join(repo_root(), path)


def build_datasets(cfg, for_teacher: bool = False) -> Tuple[Dict[str, VideoReIDDataset], Dict]:
    """Build the train / query / gallery datasets.

    Args:
        cfg: the full config.
        for_teacher: when True, build the stage-0 GaitGL dataset instead: the
            train split only, silhouettes required, no distorted twin (the
            teacher never sees RGB, so none of the bias machinery applies).

    Returns ``(datasets, meta)`` where ``meta`` carries the label-space sizes
    the model needs (``num_identities``, ``num_actions``, ``multilabel``).
    """
    data_cfg = cfg.data
    annotation_path = _abs(data_cfg.annotation)
    if not os.path.exists(annotation_path):
        raise FileNotFoundError(
            f"annotation file not found: {annotation_path}\n"
            f"Build it first with tools/prepare_dataset.py (see the README's "
            f"'Reproducing the paper' section)."
        )

    annotation = load_annotation(annotation_path)
    log_dataset_summary(annotation)

    data_root = _abs(data_cfg.get("data_root", "data"))
    num_actions = int(annotation.get("num_actions", 1))
    multilabel = bool(annotation.get("multilabel", False))
    splits = annotation["splits"]

    train_samples = splits.get("train", [])
    pid_map = build_pid_map(train_samples) if train_samples else {}

    # Silhouettes are a training-time signal only. Section 3: "In our proposed
    # method we only use silhouette during training and it is not required for
    # inference."
    want_silhouettes = for_teacher or bool(cfg.get("teacher", {}).get("enabled", True))
    want_distorted = (not for_teacher) and float(
        cfg.get("loss", {}).get("distortion_weight", 0.01)
    ) > 0

    common = dict(
        data_root=data_root,
        num_sampled_frames=data_cfg.get("num_frames", 8),
        frame_stride=data_cfg.get("frame_stride", 4),
        sampling=data_cfg.get("sampling", "stride"),
        frame_template=data_cfg.get("frame_template", "frame{:06d}.jpg"),
        silhouette_template=data_cfg.get("silhouette_template", "frame{:06d}.png"),
        frame_index_base=data_cfg.get("frame_index_base", 1),
        num_actions=num_actions,
        multilabel=multilabel,
    )

    datasets: Dict[str, VideoReIDDataset] = {}
    if train_samples:
        datasets["train"] = VideoReIDDataset(
            samples=train_samples,
            transform=build_transform(data_cfg, train=True),
            mode="train",
            pid_map=pid_map,
            load_silhouettes=want_silhouettes,
            make_distorted=want_distorted,
            **common,
        )
        missing = sum(1 for s in train_samples if not s.get("silhouettes_dir"))
        if want_silhouettes and missing:
            get_logger().warning(
                f"{missing}/{len(train_samples)} train clips have no 'silhouettes_dir'; "
                f"their distillation loss will be masked out. Run "
                f"tools/extract_silhouettes.py and rebuild the annotation to fix this."
            )

    if not for_teacher:
        for split in ("query", "gallery"):
            if splits.get(split):
                datasets[split] = VideoReIDDataset(
                    samples=splits[split],
                    transform=build_transform(data_cfg, train=False),
                    mode="eval",
                    pid_map=None,  # retrieval matches on raw identity labels
                    load_silhouettes=False,  # inference is RGB-only
                    make_distorted=False,
                    **common,
                )

    meta = {
        "name": annotation.get("name", "unknown"),
        "protocol": annotation.get("protocol", "same_activity"),
        "num_identities": max(len(pid_map), 1),
        "num_actions": num_actions,
        "multilabel": multilabel,
        "action_names": annotation.get("action_names", []),
        "pid_map": pid_map,
    }
    return datasets, meta


def build_train_loader(cfg, dataset: VideoReIDDataset) -> DataLoader:
    """Identity-balanced train loader.

    The paper trains with "a batch size of 32 with each batch containing 8
    person and 4 clips for each person", which is what makes the batch-hard
    triplet term of Eq. 4 well-posed.
    """
    run_cfg = cfg.run
    sampler_cfg = cfg.get("sampler", {}) or {}
    num_instances = sampler_cfg.get("num_instances", 4)
    batch_size = run_cfg.get("batch_size", 32)
    world_size = get_world_size()

    pids = [dataset.mapped_pid(s["pid"]) for s in dataset.samples]

    if is_dist_avail_and_initialized() and world_size > 1:
        if batch_size % world_size != 0:
            raise ValueError(
                f"run.batch_size ({batch_size}) must be divisible by world_size ({world_size})"
            )
        per_rank = batch_size // world_size
        sampler = DistributedRandomIdentitySampler(
            pids,
            batch_size=per_rank,
            num_instances=num_instances,
            seed=run_cfg.get("seed", 42),
        )
        loader_batch_size = per_rank
    else:
        sampler = RandomIdentitySampler(
            pids,
            batch_size=batch_size,
            num_instances=num_instances,
            seed=run_cfg.get("seed", 42),
        )
        loader_batch_size = batch_size

    get_logger().info(
        f"train loader: batch={loader_batch_size}/rank x {world_size} ranks, "
        f"K={num_instances} clips per identity, {len(sampler)} samples/epoch/rank"
    )

    num_workers = run_cfg.get("num_workers", 8)
    return DataLoader(
        dataset,
        batch_size=loader_batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=run_cfg.get("pin_memory", True),
        drop_last=True,
        collate_fn=collate_video_batch,
        persistent_workers=bool(num_workers) and run_cfg.get("persistent_workers", False),
        worker_init_fn=seed_worker,
        prefetch_factor=run_cfg.get("prefetch_factor", 2) if num_workers else None,
    )


def build_eval_loader(cfg, dataset: VideoReIDDataset) -> DataLoader:
    run_cfg = cfg.run
    batch_size = run_cfg.get("batch_size_eval", run_cfg.get("batch_size", 32))
    num_workers = run_cfg.get("num_workers", 8)

    sampler = None
    if is_dist_avail_and_initialized() and get_world_size() > 1:
        sampler = InferenceSampler(len(dataset))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=run_cfg.get("pin_memory", True),
        drop_last=False,
        collate_fn=collate_video_batch,
        persistent_workers=bool(num_workers) and run_cfg.get("persistent_workers", False),
        prefetch_factor=run_cfg.get("prefetch_factor", 2) if num_workers else None,
    )
