"""Official dataset splits and the ABNet gallery/probe protocol.

ABNet follows the evaluation setup of ABNet, "Activity-Biometrics: Person
Identification from Daily Activities" (CVPR 2024,
`arXiv:2403.17360 <https://arxiv.org/abs/2403.17360>`_), which defines the
activity-biometrics benchmarks. Two separate decisions are involved.

**1. Train / test separation** uses each dataset's own official split:

* NTU RGB-AB -- "We use the official cross-subject split for the train test
  separation." 106 subjects, so this is the NTU RGB+D **120** X-Sub split.
* Charades-AB -- "We use the official train-test split for our experiments."
* Charades-AB -- "We use the official train-test split for our experiments."

**2. Gallery / probe construction** then splits the *test* portion
(ABNet Appendix A, quoted):

    "In the same activity evaluation protocol, probe and gallery contains all
    the activities, however, probe contains a smaller subset of samples and the
    rest are placed in gallery. In the cross activity evaluation protocol,
    probe and gallery contains mutually exclusive activities, where probe
    contains a smaller subset of samples and rest of the samples from those
    activities are discarded; on the contrary the gallery contains all samples
    from a certain activity. ... The samples are randomly selected for gallery
    and probe sets."

The View+ / View- distinction ("probe view included in gallery" vs "probe view
excluded from gallery") is *not* part of the split: it is applied at retrieval
time by the gallery filter in :mod:`abnet.evaluation.metrics`.
"""

import random
from collections import defaultdict
from typing import Dict, List, Sequence, Set

#: NTU RGB+D 120 cross-subject (X-Sub) training subject IDs, from the NTU
#: RGB+D 120 release (Liu et al., TPAMI 2019). 53 of the 106 subjects train,
#: the other 53 test. NTU RGB-AB reports 106 actors, so this is the right list.
NTU120_XSUB_TRAIN_SUBJECTS: Set[int] = {
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38,
    45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78, 80, 81, 82,
    83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103,
}

#: NTU RGB+D 60 X-Sub training subjects, for the 60-class subset.
NTU60_XSUB_TRAIN_SUBJECTS: Set[int] = {
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38,
}

#: NTU RGB+D two-person ("mutual") action IDs. Excluding them leaves the 94
#: single-person classes that NTU RGB-AB reports (120 - 26 = 94).
NTU_MUTUAL_ACTIONS: Set[int] = set(range(50, 61)) | set(range(106, 121))

def action_set(sample: Dict) -> Set[int]:
    """Activity labels of a sample as a set (handles the multi-label case)."""
    value = sample.get("action")
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {int(a) for a in value}
    return {int(value)}


def build_gallery_probe(
    test_samples: Sequence[Dict],
    protocol: str = "same_activity",
    probe_ratio: float = 0.25,
    seed: int = 42,
) -> Dict[str, List[Dict]]:
    """Split the test portion into probe and gallery, per ABNet Appendix A.

    Args:
        test_samples: every clip belonging to the test identities.
        protocol: ``same_activity`` or ``cross_activity``.
        probe_ratio: the "smaller subset" that becomes the probe.
        seed: controls the random selection, which ABNet leaves random.

    Returns:
        ``{"query": [...], "gallery": [...], "discarded": [...]}``.
    """
    if protocol not in ("same_activity", "cross_activity"):
        raise ValueError(
            f"protocol must be 'same_activity' or 'cross_activity', got '{protocol}'"
        )
    rng = random.Random(seed)
    samples = list(test_samples)
    rng.shuffle(samples)

    if protocol == "same_activity":
        # Both sets hold all activities; the probe is a small random subset.
        # Selected per identity so every identity is represented in both.
        by_pid = defaultdict(list)
        for sample in samples:
            by_pid[sample["pid"]].append(sample)

        query, gallery = [], []
        for pid_samples in by_pid.values():
            if len(pid_samples) < 2:
                # A lone clip cannot be both probe and gallery; keep it as
                # gallery so it still acts as a distractor.
                gallery.extend(pid_samples)
                continue
            count = max(1, min(int(round(len(pid_samples) * probe_ratio)),
                               len(pid_samples) - 1))
            query.extend(pid_samples[:count])
            gallery.extend(pid_samples[count:])
        return {"query": query, "gallery": gallery, "discarded": []}

    # ---- cross activity ----
    # Activities are partitioned into two mutually exclusive groups. The probe
    # draws a small subset from its group and the remaining samples of those
    # activities are discarded; the gallery keeps every sample of its group.
    activities = sorted({a for s in samples for a in action_set(s)})
    if len(activities) < 2:
        raise ValueError(
            "cross_activity needs at least 2 activity classes in the test split"
        )
    shuffled = list(activities)
    rng.shuffle(shuffled)
    probe_activities = set(shuffled[: max(len(shuffled) // 2, 1)])

    # Multi-label clips (Charades-AB) are assigned by majority membership.
    #
    # Note what this can and cannot guarantee. For single-label datasets the
    # probe and gallery activity sets come out exactly disjoint, which is the
    # literal reading of the protocol. For Charades-AB it is not achievable:
    # a video carries 6.8 of 157 activities on average, so under any 50/50
    # partition of the activity list essentially every video contains both a
    # probe-group and a gallery-group activity. Demanding strict per-label
    # exclusivity would discard almost the entire benchmark, which is plainly
    # not what the paper does -- it reports Charades-AB cross-activity numbers.
    #
    # So each clip is assigned to exactly one side by majority, and the *unions*
    # of their label sets still overlap. Per-query activity exclusion is then
    # enforced at match time by the ``cross_activity`` gallery filter in
    # :mod:`abnet.evaluation.metrics`, which drops gallery clips sharing the
    # probe's primary activity. That two-level arrangement is the only
    # consistent way to read the protocol for a multi-label dataset.
    def is_probe_side(sample):
        labels = action_set(sample)
        if not labels:
            return False
        return len(labels & probe_activities) * 2 >= len(labels)

    probe_pool = [s for s in samples if is_probe_side(s)]
    gallery = [s for s in samples if not is_probe_side(s)]

    by_pid = defaultdict(list)
    for sample in probe_pool:
        by_pid[sample["pid"]].append(sample)

    gallery_pids = {s["pid"] for s in gallery}
    query, discarded = [], []
    for pid, pid_samples in by_pid.items():
        if pid not in gallery_pids:
            # No gallery match exists for this identity, so it cannot be scored.
            discarded.extend(pid_samples)
            continue
        count = max(1, int(round(len(pid_samples) * probe_ratio)))
        query.extend(pid_samples[:count])
        discarded.extend(pid_samples[count:])

    return {"query": query, "gallery": gallery, "discarded": discarded}


def enforce_disjoint_identities(
    train: Sequence[Dict], test: Sequence[Dict]
) -> Dict[str, List[Dict]]:
    """Drop test identities that also appear in train.

    Person identification requires disjoint identity sets. The official
    *action-recognition* splits of some datasets are video-level rather than
    actor-level, so this guards the re-ID protocol. ABNet notes that for the
    datasets it uses "the test and train split contains mutually exclusive
    actors/subjects", so in practice this should remove little or nothing.
    """
    train_pids = {str(s["pid"]) for s in train}
    kept = [s for s in test if str(s["pid"]) not in train_pids]
    removed = [s for s in test if str(s["pid"]) in train_pids]
    return {"test": kept, "removed": removed}
