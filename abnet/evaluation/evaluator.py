"""Feature extraction and retrieval/verification evaluation.

The distance matrix is computed **once** and reused for every protocol, because
the protocols differ only in which gallery entries are admitted at match time,
not in the features. For NTU RGB-AB that turns four reported protocol variants
into one pass over the data instead of four.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from abnet.evaluation.metrics import (
    compute_cmc_map,
    compute_tar_at_far,
    format_results,
    summary_dict,
)
from abnet.utils.dist import all_gather_tensor, get_world_size
from abnet.utils.logger import MetricLogger, get_logger
from abnet.utils.misc import unwrap_model

#: Protocol variants reported per dataset.
#: ``junk_rule`` filters the gallery; ``cross_activity`` additionally removes
#: same-activity gallery clips at match time.
PROTOCOLS = {
    # NTU RGB-AB: viewpoint is recorded, so View+/View- apply.
    "same_activity_include_view": {"junk_rule": "none", "cross_activity": False},
    "same_activity_exclude_view": {"junk_rule": "same_view", "cross_activity": False},
    "cross_activity_include_view": {"junk_rule": "none", "cross_activity": True},
    "cross_activity_exclude_view": {"junk_rule": "same_view", "cross_activity": True},
    # Charades-AB: "since Charades ... does not contain multiple view points,
    # the evaluation protocol with inclusion/exclusion of probe view from
    # gallery is not relevant in these case."
    "same_activity": {"junk_rule": "none", "cross_activity": False},
    "cross_activity": {"junk_rule": "none", "cross_activity": True},
    # Conventional video re-ID rule, for comparison against other benchmarks.
    "general": {"junk_rule": "same_pid_same_view", "cross_activity": False},
}


@torch.no_grad()
def extract_split_features(
    model,
    loader,
    device: torch.device,
    amp_dtype: Optional[torch.dtype] = None,
    print_freq: int = 50,
    header: str = "extract",
    feature: str = "fused",
) -> Dict[str, np.ndarray]:
    """Run the inference path over a loader and gather features across ranks.

    Returns numpy arrays for ``feature``, ``pid``, ``view`` and ``action``,
    ordered by dataset index with any distributed padding removed.
    """
    net = unwrap_model(model)
    net.eval()

    chunks: Dict[str, List[torch.Tensor]] = {
        key: [] for key in ("feature", "pid", "view", "action", "index")
    }

    metric_logger = MetricLogger()
    for _, batch in metric_logger.log_every(loader, print_freq, header=header):
        frames = batch["frames"].to(device, non_blocking=True)

        autocast = (
            torch.autocast(device_type=device.type, dtype=amp_dtype)
            if amp_dtype is not None and device.type == "cuda"
            else torch.autocast(device_type="cpu", enabled=False)
        )
        with autocast:
            vector = net.extract_features(frames, feature=feature, normalize=True)

        chunks["feature"].append(vector.float())
        chunks["pid"].append(batch["raw_pid"].to(device))
        chunks["view"].append(batch["view"].to(device))
        chunks["action"].append(batch["action_id"].to(device))
        chunks["index"].append(batch["index"].to(device))

    merged = {key: torch.cat(values, dim=0) for key, values in chunks.items()}

    if get_world_size() > 1:
        merged = {key: all_gather_tensor(value) for key, value in merged.items()}

    # Undo the InferenceSampler padding and restore dataset order.
    index = merged.pop("index").cpu().numpy()
    _, unique_positions = np.unique(index, return_index=True)
    order = unique_positions[np.argsort(index[unique_positions])]

    return {key: value.cpu().numpy()[order] for key, value in merged.items()}


def cosine_distance_matrix(
    query: np.ndarray, gallery: np.ndarray, chunk_size: int = 2048
) -> np.ndarray:
    """``1 - cosine`` between L2-normalised features, computed in row chunks.

    Chunking keeps the peak allocation bounded: NTU RGB-AB's gallery is large
    enough that a single dense product is wasteful even though the result
    itself must be materialised.
    """
    distances = np.empty((query.shape[0], gallery.shape[0]), dtype=np.float32)
    for start in range(0, query.shape[0], chunk_size):
        stop = min(start + chunk_size, query.shape[0])
        distances[start:stop] = 1.0 - query[start:stop] @ gallery.T
    return distances


def evaluate(
    model,
    query_loader,
    gallery_loader,
    device: torch.device,
    protocols: Sequence[str],
    amp_dtype: Optional[torch.dtype] = None,
    max_rank: int = 50,
    print_freq: int = 50,
    feature: str = "fused",
    far_targets: Sequence[float] = (0.001,),
) -> Dict[str, Dict[str, float]]:
    """Extract both splits and report every requested protocol.

    Args:
        protocols: keys of :data:`PROTOCOLS`.
        feature: which representation to retrieve with. ``fused`` is the
            paper's ``concat(F_Ac, f_bb)``; ``biometrics`` drops the activity
            prior; ``appearance`` should score *poorly*, which is the evidence
            that disentanglement worked.
        far_targets: FAR operating points for TAR; 0.001 is the paper's.
    """
    logger = get_logger()
    unknown = [p for p in protocols if p not in PROTOCOLS]
    if unknown:
        raise ValueError(f"unknown protocol(s) {unknown}; available: {sorted(PROTOCOLS)}")

    query = extract_split_features(
        model, query_loader, device, amp_dtype, print_freq, "query  ", feature
    )
    gallery = extract_split_features(
        model, gallery_loader, device, amp_dtype, print_freq, "gallery", feature
    )
    logger.info(
        f"extracted '{feature}' features: query={query['feature'].shape}, "
        f"gallery={gallery['feature'].shape}"
    )

    distmat = cosine_distance_matrix(query["feature"], gallery["feature"])

    shared = dict(
        query_pids=query["pid"],
        gallery_pids=gallery["pid"],
        query_views=query["view"],
        gallery_views=gallery["view"],
        query_actions=query["action"],
        gallery_actions=gallery["action"],
    )

    results: Dict[str, Dict[str, float]] = {}
    for name in protocols:
        spec = PROTOCOLS[name]
        raw = compute_cmc_map(
            distmat=distmat,
            max_rank=max_rank,
            junk_rule=spec["junk_rule"],
            cross_activity_runtime=spec["cross_activity"],
            **shared,
        )
        raw.update(
            compute_tar_at_far(
                distmat=distmat,
                junk_rule=spec["junk_rule"],
                cross_activity_runtime=spec["cross_activity"],
                far_targets=far_targets,
                **shared,
            )
        )
        logger.info(f"[{feature}] {name:>30}  {format_results(raw)}")
        results[name] = summary_dict(raw)
    return results
