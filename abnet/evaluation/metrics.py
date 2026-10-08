"""Retrieval and verification metrics for the activity-biometrics protocols.

The paper reports four metrics: "we employ rank 1 accuracy, rank 5 accuracy,
mean average precision (mAP), and TAR @ 0.1% FAR. While the first three
evaluation metrics are more popular to evaluate a person identification model,
the latter metric is also crucial to check the model's ability to minimize the
false acceptance rate."

Section 4 and Appendix A define the protocol axes:

activity
    ``same``   -- all activities appear in both probe and gallery.
    ``cross``  -- activities in one set are absent from the other. This is a
                  property of the split, but ``cross_activity_runtime=True``
                  can also enforce it at match time by discarding gallery
                  clips whose activity matches the probe's.

view (NTU RGB-AB only, which records viewpoint)
    ``include`` -- every viewpoint is available in the gallery (View+).
    ``exclude`` -- the probe's own viewpoint is removed from the gallery (View-).

``junk_rule="same_pid_same_view"`` is the standard video re-ID rule (drop
gallery clips sharing both identity and camera with the probe), kept for
comparison against conventional benchmarks.
"""

from typing import Dict, Optional, Sequence

import numpy as np

#: Gallery-filtering rules.
JUNK_RULES = ("none", "same_view", "same_pid_same_view")


def compute_cmc_map(
    distmat: np.ndarray,
    query_pids: np.ndarray,
    gallery_pids: np.ndarray,
    query_views: Optional[np.ndarray] = None,
    gallery_views: Optional[np.ndarray] = None,
    query_actions: Optional[np.ndarray] = None,
    gallery_actions: Optional[np.ndarray] = None,
    max_rank: int = 50,
    junk_rule: str = "none",
    cross_activity_runtime: bool = False,
) -> Dict[str, object]:
    """Rank-k accuracies and mean average precision.

    Args:
        distmat: ``[num_query, num_gallery]`` distances (smaller = more similar).
        query_pids / gallery_pids: identity labels.
        query_views / gallery_views: camera or viewpoint ids, needed for any
            ``junk_rule`` other than ``"none"``.
        query_actions / gallery_actions: activity ids, needed when
            ``cross_activity_runtime`` is set.
        junk_rule: one of :data:`JUNK_RULES`.
        cross_activity_runtime: also discard same-activity gallery tracklets.

    Returns:
        ``{"cmc": array[max_rank], "mAP": float, "num_valid_query": int}``.
    """
    if junk_rule not in JUNK_RULES:
        raise ValueError(f"junk_rule must be one of {JUNK_RULES}, got '{junk_rule}'")
    if junk_rule != "none" and (query_views is None or gallery_views is None):
        raise ValueError(f"junk_rule='{junk_rule}' needs query_views and gallery_views")
    if cross_activity_runtime and (query_actions is None or gallery_actions is None):
        raise ValueError("cross_activity_runtime needs query_actions and gallery_actions")

    num_query, num_gallery = distmat.shape
    max_rank = min(max_rank, num_gallery)

    indices = np.argsort(distmat, axis=1)
    matches = (gallery_pids[indices] == query_pids[:, None]).astype(np.int32)

    all_cmc, all_ap = [], []
    num_valid_query = 0

    for q in range(num_query):
        order = indices[q]
        keep_mask = gallery_keep_mask(
            q, query_pids, gallery_pids, query_views, gallery_views,
            query_actions, gallery_actions, junk_rule, cross_activity_runtime,
        )
        keep = keep_mask[order]
        row_matches = matches[q][keep]
        if not row_matches.any():
            # No ground-truth match survives the filter: this probe is skipped,
            # exactly as in the standard Market-1501 / MARS evaluation.
            continue

        cmc = row_matches.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])

        num_relevant = row_matches.sum()
        cumulative = row_matches.cumsum()
        precision = cumulative / (np.arange(len(row_matches)) + 1.0)
        all_ap.append((precision * row_matches).sum() / num_relevant)
        num_valid_query += 1

    if num_valid_query == 0:
        raise RuntimeError(
            "no valid query remained after gallery filtering; check that query and "
            "gallery share identities and that the junk_rule is appropriate"
        )

    padded = np.zeros((len(all_cmc), max_rank), dtype=np.float32)
    for i, cmc in enumerate(all_cmc):
        padded[i, : len(cmc)] = cmc
        if len(cmc) < max_rank:
            padded[i, len(cmc) :] = cmc[-1]

    return {
        "cmc": padded.mean(axis=0),
        "mAP": float(np.mean(all_ap)),
        "num_valid_query": num_valid_query,
    }


def gallery_keep_mask(
    query_index: int,
    query_pids: np.ndarray,
    gallery_pids: np.ndarray,
    query_views: Optional[np.ndarray] = None,
    gallery_views: Optional[np.ndarray] = None,
    query_actions: Optional[np.ndarray] = None,
    gallery_actions: Optional[np.ndarray] = None,
    junk_rule: str = "none",
    cross_activity_runtime: bool = False,
) -> np.ndarray:
    """Boolean mask of gallery entries usable for one probe.

    Factored out so :func:`compute_cmc_map` and :func:`compute_tar_at_far`
    apply byte-identical filtering; a divergence between the two would make the
    reported rank-1 and TAR@FAR describe different experiments.
    """
    remove = np.zeros(len(gallery_pids), dtype=bool)
    if junk_rule == "same_view":
        remove |= gallery_views == query_views[query_index]
    elif junk_rule == "same_pid_same_view":
        remove |= (gallery_pids == query_pids[query_index]) & (
            gallery_views == query_views[query_index]
        )
    if cross_activity_runtime:
        remove |= gallery_actions == query_actions[query_index]
    return ~remove


def compute_tar_at_far(
    distmat: np.ndarray,
    query_pids: np.ndarray,
    gallery_pids: np.ndarray,
    query_views: Optional[np.ndarray] = None,
    gallery_views: Optional[np.ndarray] = None,
    query_actions: Optional[np.ndarray] = None,
    gallery_actions: Optional[np.ndarray] = None,
    junk_rule: str = "none",
    cross_activity_runtime: bool = False,
    far_targets: Sequence[float] = (0.001,),
) -> Dict[str, float]:
    """True acceptance rate at fixed false acceptance rates.

    This is the 1:1 *verification* view of the same probe-gallery matrix that
    CMC scores for retrieval. Every surviving (probe, gallery) pair is one
    comparison: genuine when the identities match, impostor otherwise. For each
    target FAR a threshold is chosen on the impostor score distribution, and
    TAR is the fraction of genuine pairs scoring at or above it.

    Scores are negated distances, so larger is more similar.

    Args:
        far_targets: e.g. ``(0.001,)`` for the paper's TAR @ 0.1% FAR.

    Returns:
        ``{"TAR@0.1%FAR": 97.3, ...}`` in percent, plus the pair counts.
    """
    if junk_rule not in JUNK_RULES:
        raise ValueError(f"junk_rule must be one of {JUNK_RULES}, got '{junk_rule}'")

    genuine, impostor = [], []
    for q in range(distmat.shape[0]):
        keep = gallery_keep_mask(
            q, query_pids, gallery_pids, query_views, gallery_views,
            query_actions, gallery_actions, junk_rule, cross_activity_runtime,
        )
        if not keep.any():
            continue
        scores = -distmat[q][keep]
        same = gallery_pids[keep] == query_pids[q]
        genuine.append(scores[same])
        impostor.append(scores[~same])

    genuine_scores = np.concatenate(genuine) if genuine else np.empty(0)
    impostor_scores = np.concatenate(impostor) if impostor else np.empty(0)

    out: Dict[str, float] = {
        "num_genuine_pairs": int(genuine_scores.size),
        "num_impostor_pairs": int(impostor_scores.size),
    }
    if genuine_scores.size == 0 or impostor_scores.size == 0:
        for far in far_targets:
            out[_far_key(far)] = float("nan")
        return out

    impostor_sorted = np.sort(impostor_scores)[::-1]
    for far in far_targets:
        # Threshold admitting at most `far` of the impostor comparisons. With
        # few impostor pairs the requested FAR may be unreachable; taking the
        # top-1 impostor score is then the strictest achievable operating
        # point, and the realised FAR is reported alongside so the limitation
        # is visible rather than silent.
        count = int(np.floor(far * impostor_sorted.size))
        position = min(max(count, 1), impostor_sorted.size) - 1
        threshold = impostor_sorted[position]
        out[_far_key(far)] = 100.0 * float((genuine_scores >= threshold).mean())
        out[_far_key(far) + "_realised_far"] = 100.0 * float(
            (impostor_scores >= threshold).mean()
        )
    return out


def _far_key(far: float) -> str:
    """``0.001 -> 'TAR@0.1%FAR'``."""
    percent = far * 100.0
    text = f"{percent:g}"
    return f"TAR@{text}%FAR"


def format_results(results: Dict[str, object], ranks=(1, 5, 10, 20)) -> str:
    cmc = results["cmc"]
    parts = [f"mAP: {100 * results['mAP']:.2f}"]
    for rank in ranks:
        if rank <= len(cmc):
            parts.append(f"R@{rank}: {100 * cmc[rank - 1]:.2f}")
    for key, value in results.items():
        if isinstance(key, str) and key.startswith("TAR@") and not key.endswith("_realised_far"):
            parts.append(f"{key}: {value:.2f}")
    parts.append(f"valid_query: {results['num_valid_query']}")
    return "  ".join(parts)


def summary_dict(results: Dict[str, object], ranks=(1, 5, 10, 20)) -> Dict[str, float]:
    cmc = results["cmc"]
    out = {"mAP": 100 * float(results["mAP"])}
    for rank in ranks:
        if rank <= len(cmc):
            out[f"R@{rank}"] = 100 * float(cmc[rank - 1])
    for key, value in results.items():
        if isinstance(key, str) and key.startswith("TAR@"):
            out[key] = float(value)
    out["num_valid_query"] = int(results["num_valid_query"])
    return out
