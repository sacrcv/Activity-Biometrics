#!/usr/bin/env python
"""Build the annotation JSON for an activity-biometrics benchmark.

Implements the dataset construction of ABNet (`arXiv:2403.17360
<https://arxiv.org/abs/2403.17360>`_): each dataset's **official** train/test
split, then a random probe/gallery division of the test portion (Appendix A).
See :mod:`abnet.data.splits` for the exact rules and the quoted protocol text.

Pass ``--silhouettes-root`` to record a ``silhouettes_dir`` per clip, which is
what lets the teacher train and the distillation loss fire. Clips with no
silhouette directory are still included -- their ``L_KD`` term is masked out --
so you can build an annotation before extraction finishes.

Datasets
--------
``ntu`` -- **NTU RGB-AB**, built from NTU RGB+D 120.
    Clip metadata comes from the ``SsssCcccPpppRrrrAaaa`` directory names
    (setup, camera, performer, replication, action). The 26 two-person actions
    are dropped, leaving the 94 single-person classes and 106 performers the
    papers report. Train/test uses the official NTU-120 **cross-subject** list.

``charades`` -- **Charades-AB**.
    Reads the official ``Charades_v1_train.csv`` / ``Charades_v1_test.csv``;
    ``subject`` is the actor (267 of them) and ``actions`` gives the 157
    multi-label classes.

``manifest`` -- any dataset, from a CSV you provide.
    Required columns ``id``, ``pid``, ``action`` and one of
    ``frames_dir``/``video``; optional ``num_frames``, ``view``, ``keyframe``,
    ``outfit``, ``split``.

Examples
--------
::

    python tools/prepare_dataset.py --dataset ntu \\
        --frames-root data/ntu_rgb_ab/frames \\
        --silhouettes-root data/ntu_rgb_ab/silhouettes \\
        --out data/ntu_rgb_ab/annotations.json --protocol same_activity

    python tools/prepare_dataset.py --dataset charades \\
        --charades-csv-dir data/charades_ab/annotations \\
        --frames-root data/charades_ab/frames \\
        --silhouettes-root data/charades_ab/silhouettes \\
        --out data/charades_ab/annotations.json

Produce one annotation file per protocol; ``--protocol cross_activity`` writes
the mutually-exclusive-activity variant. Both are needed to report the paper's
full protocol table.
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abnet.data.splits import (  # noqa: E402
    NTU60_XSUB_TRAIN_SUBJECTS,
    NTU120_XSUB_TRAIN_SUBJECTS,
    NTU_MUTUAL_ACTIONS,
    build_gallery_probe,
    enforce_disjoint_identities,
)

NTU_PATTERN = re.compile(
    r"S(?P<setup>\d{3})C(?P<camera>\d{3})P(?P<performer>\d{3})"
    r"R(?P<replication>\d{3})A(?P<action>\d{3})",
    re.IGNORECASE,
)
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def count_frames(path):
    if not os.path.isdir(path):
        return 0
    return sum(1 for n in os.listdir(path) if n.lower().endswith(IMAGE_SUFFIXES))


def parse_int_set(spec):
    """Parse ``"42-51,60"`` into ``{42,...,51,60}``."""
    if not spec:
        return set()
    out = set()
    for piece in str(spec).split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            lo, hi = piece.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(piece))
    return out


# ----------------------------------------------------------------------
# scanners: native layout -> flat records + an official train/test tag
# ----------------------------------------------------------------------
def scan_ntu(args):
    root = args.frames_root
    if not root or not os.path.isdir(root):
        raise FileNotFoundError(f"--frames-root not found: {root}")

    train_subjects = (
        NTU60_XSUB_TRAIN_SUBJECTS if args.ntu_version == 60
        else NTU120_XSUB_TRAIN_SUBJECTS
    )
    exclude = parse_int_set(args.exclude_actions) or NTU_MUTUAL_ACTIONS
    print(f"  NTU-{args.ntu_version} official cross-subject split: "
          f"{len(train_subjects)} training subjects")
    print(f"  excluding {len(exclude)} two-person action classes")

    records, skipped = [], 0
    for name in sorted(os.listdir(root)):
        match = NTU_PATTERN.search(name)
        if not match:
            continue
        action = int(match.group("action"))
        if action in exclude:
            continue
        frames_dir = os.path.join(root, name)
        if not os.path.isdir(frames_dir):
            continue
        num_frames = count_frames(frames_dir)
        if num_frames < args.min_frames:
            skipped += 1
            continue
        performer = int(match.group("performer"))
        records.append({
            "id": name,
            "frames_dir": os.path.relpath(frames_dir, args.data_root),
            "num_frames": num_frames,
            "pid": performer,
            "action": action,
            "view": int(match.group("camera")),
            "setup": int(match.group("setup")),
            "keyframe": num_frames // 2,
            "_split": "train" if performer in train_subjects else "test",
        })
    if skipped:
        print(f"  skipped {skipped} clips with < {args.min_frames} frames")
    return records, None, False


def scan_charades(args):
    if not args.charades_csv_dir or not os.path.isdir(args.charades_csv_dir):
        raise FileNotFoundError(
            "--charades-csv-dir is required (holding Charades_v1_train.csv and "
            "Charades_v1_test.csv)"
        )

    action_ids, records = {}, []
    for split, tag in (("train", "train"), ("test", "test")):
        csv_path = os.path.join(args.charades_csv_dir, f"Charades_v1_{split}.csv")
        if not os.path.exists(csv_path):
            print(f"  note: {csv_path} not found, skipping")
            continue
        count = 0
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                video_id = row["id"]
                frames_dir = os.path.join(args.frames_root or "", video_id)
                num_frames = count_frames(
                    frames_dir if os.path.isabs(frames_dir)
                    else os.path.join(args.data_root, os.path.relpath(frames_dir, args.data_root))
                    if args.frames_root else frames_dir
                )
                if num_frames < args.min_frames:
                    continue

                actions = []
                for entry in (row.get("actions") or "").split(";"):
                    entry = entry.strip()
                    if entry:
                        actions.append(action_ids.setdefault(entry.split()[0], len(action_ids)))
                if not actions:
                    continue

                records.append({
                    "id": video_id,
                    "frames_dir": os.path.relpath(frames_dir, args.data_root),
                    "num_frames": num_frames,
                    "pid": row["subject"],
                    "action": sorted(set(actions)),
                    "view": 0,          # Charades has no viewpoint annotation
                    "keyframe": num_frames // 2,
                    "_split": tag,
                })
                count += 1
        print(f"  official {split} split: {count} videos")

    names = [None] * len(action_ids)
    for class_name, index in action_ids.items():
        names[index] = class_name
    return records, names, True


def scan_manifest(args):
    if not args.manifest or not os.path.exists(args.manifest):
        raise FileNotFoundError("--manifest is required for --dataset manifest")

    records, action_names, multilabel = [], None, False
    with open(args.manifest, newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        missing = {"id", "pid", "action"} - fields
        if missing:
            raise ValueError(f"manifest is missing required column(s): {sorted(missing)}")
        if not ({"frames_dir", "video"} & fields):
            raise ValueError("manifest needs a 'frames_dir' or a 'video' column")

        for row in reader:
            action_field = (row.get("action") or "").strip()
            if re.search(r"[;,]", action_field):
                actions = sorted({int(a) for a in re.split(r"[;,]", action_field) if a.strip()})
                multilabel = True
            else:
                actions = int(action_field)

            frames_dir = (row.get("frames_dir") or "").strip() or None
            num_frames = int(row["num_frames"]) if (row.get("num_frames") or "").strip() else 0
            if frames_dir and not num_frames:
                probe = (frames_dir if os.path.isabs(frames_dir)
                         else os.path.join(args.data_root, frames_dir))
                num_frames = count_frames(probe)
            if num_frames and num_frames < args.min_frames:
                continue

            records.append({
                "id": row["id"],
                "frames_dir": frames_dir,
                "video": (row.get("video") or "").strip() or None,
                "num_frames": num_frames,
                "pid": row["pid"],
                "action": actions,
                "view": int(row.get("view") or 0),
                **({"outfit": int(row["outfit"])} if (row.get("outfit") or "").strip() else {}),
                "keyframe": int(row.get("keyframe") or (num_frames // 2)),
                "_split": (row.get("split") or "").strip() or None,
            })

    if all(r["_split"] is None for r in records):
        subjects = sorted({str(r["pid"]) for r in records})
        train_subjects = set(subjects[: max(int(len(subjects) * args.train_identity_ratio), 1)])
        for record in records:
            record["_split"] = "train" if str(record["pid"]) in train_subjects else "test"
        print(f"  no 'split' column; cross-subject split by actor "
              f"({len(train_subjects)}/{len(subjects)} actors train)")

    if args.action_names and os.path.exists(args.action_names):
        action_names = [line.strip() for line in open(args.action_names) if line.strip()]
    return records, action_names, multilabel


SCANNERS = {
    "ntu": scan_ntu,
    "charades": scan_charades,
    "manifest": scan_manifest,
}


def normalise_labels(records):
    """Remap identity and action labels to contiguous integers."""
    pids = sorted({str(r["pid"]) for r in records})
    pid_map = {pid: i for i, pid in enumerate(pids)}

    actions = set()
    for record in records:
        value = record["action"]
        actions.update(value if isinstance(value, list) else [value])
    action_map = {a: i for i, a in enumerate(sorted(actions))}

    for record in records:
        record["pid"] = pid_map[str(record["pid"])]
        value = record["action"]
        record["action"] = (
            [action_map[a] for a in value] if isinstance(value, list) else action_map[value]
        )
    return pid_map, action_map


def attach_silhouettes(records, args):
    """Record a ``silhouettes_dir`` for every clip that has one on disk.

    ``tools/extract_silhouettes.py`` mirrors the *frames* layout, one output
    directory per input frame directory, so the silhouette directory is found
    from the basename of ``frames_dir``. Clips that share a frame directory and
    are distinguished only by ``frame_offset`` therefore work too, since the
    same offset applies to the silhouettes.
    """
    if not args.silhouettes_root:
        return 0

    root = args.silhouettes_root
    found = 0
    for record in records:
        frames_dir = record.get("frames_dir")
        if not frames_dir:
            continue
        candidate = os.path.join(root, os.path.basename(frames_dir.rstrip("/")))
        probe = candidate if os.path.isabs(candidate) else os.path.join(
            args.data_root, os.path.relpath(candidate, args.data_root)
        )
        if os.path.isdir(probe):
            record["silhouettes_dir"] = os.path.relpath(candidate, args.data_root)
            found += 1
    return found


def main():
    args = parse_args()
    print(f"scanning '{args.dataset}' ...")
    scanned, action_names, multilabel = SCANNERS[args.dataset](args)

    if not scanned:
        raise SystemExit(
            "no clips found -- check --frames-root and that frames were extracted"
        )
    print(f"found {len(scanned)} clips")
    normalise_labels(scanned)

    if args.silhouettes_root:
        found = attach_silhouettes(scanned, args)
        print(f"  silhouettes found for {found}/{len(scanned)} clips")
        if found == 0:
            print(f"  warning: nothing matched under {args.silhouettes_root}. The "
                  f"teacher cannot be trained and L_KD will be masked out. Check "
                  f"that tools/extract_silhouettes.py wrote one directory per clip.")

    train = [r for r in scanned if r["_split"] == "train"]
    test = [r for r in scanned if r["_split"] == "test"]
    print(f"official split: {len(train)} train / {len(test)} test clips")
    if not train or not test:
        raise SystemExit("the official split produced an empty side")

    if args.enforce_disjoint_identities:
        result = enforce_disjoint_identities(train, test)
        if result["removed"]:
            print(f"  removed {len(result['removed'])} test clips whose actor "
                  f"also appears in train (identity sets must be disjoint)")
        test = result["test"]

    parts = build_gallery_probe(
        test, protocol=args.protocol, probe_ratio=args.probe_ratio, seed=args.seed
    )
    splits = {"train": train, "query": parts["query"], "gallery": parts["gallery"]}
    discarded = parts["discarded"]
    if discarded:
        print(f"  {len(discarded)} test clips discarded by the "
              f"{args.protocol} protocol (ABNet Appendix A)")

    num_actions = 1
    for records in splits.values():
        for record in records:
            value = record["action"]
            top = max(value) if isinstance(value, list) else value
            num_actions = max(num_actions, top + 1)
    if action_names is None:
        action_names = [f"action {i}" for i in range(num_actions)]

    annotation = {
        "name": args.name or args.dataset,
        "protocol": args.protocol,
        "source": "ABNet protocol (arXiv:2403.17360); official split + "
                  "random probe/gallery division of the test set",
        "num_actions": num_actions,
        "action_names": action_names,
        "multilabel": bool(multilabel),
        "splits": {
            name: [{k: v for k, v in r.items() if not k.startswith("_")} for r in records]
            for name, records in splits.items()
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(annotation, f, indent=2)

    print(f"\nwrote {args.out}")
    for name, records in annotation["splits"].items():
        pids = {r["pid"] for r in records}
        views = {r.get("view", 0) for r in records}
        print(f"  {name:>8}: {len(records):7d} clips | {len(pids):4d} identities | "
              f"{len(views):3d} views")
    overlap = ({r["pid"] for r in annotation["splits"].get("train", [])}
               & {r["pid"] for r in annotation["splits"].get("query", [])})
    with_sil = sum(
        1 for records in annotation["splits"].values()
        for r in records if r.get("silhouettes_dir")
    )
    print(f"  {num_actions} actions | multilabel={annotation['multilabel']} | "
          f"protocol={args.protocol} | train/test identity overlap={len(overlap)}")
    print(f"  {with_sil} clip(s) carry a silhouettes_dir")

    train_sil = sum(
        1 for r in annotation["splits"].get("train", []) if r.get("silhouettes_dir")
    )
    if train_sil:
        print("\nNext, train the bias-less teacher, then ABNet:")
        print(f"  bash scripts/train_teacher.sh configs/teacher_{args.name or args.dataset}.yaml")
        print(f"  bash scripts/train.sh configs/{args.name or args.dataset}.yaml")
    else:
        print("\nNext, extract silhouettes so the teacher can be trained:")
        print(f"  python tools/extract_silhouettes.py --frames-root {args.frames_root} \\")
        print(f"      --out {os.path.join(os.path.dirname(args.out), 'silhouettes')}")
        print("  then re-run this script with --silhouettes-root pointing at it.")


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dataset", required=True, choices=sorted(SCANNERS))
    parser.add_argument("--out", required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--data-root", default="data",
                        help="annotation paths are stored relative to this")
    parser.add_argument("--frames-root", default=None)
    parser.add_argument("--silhouettes-root", default=None,
                        help="output of tools/extract_silhouettes.py; records a "
                             "'silhouettes_dir' per clip so the teacher can train")
    parser.add_argument("--protocol", default="same_activity",
                        choices=["same_activity", "cross_activity"])
    parser.add_argument("--probe-ratio", type=float, default=0.25,
                        help="ABNet's 'smaller subset' of the test set used as probes")
    parser.add_argument("--min-frames", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude-actions", default=None,
                        help="action ids to drop, e.g. '50-60,106-120'; defaults to "
                             "each dataset's two-person classes")
    parser.add_argument("--enforce-disjoint-identities", action="store_true", default=True)
    parser.add_argument("--allow-shared-identities", dest="enforce_disjoint_identities",
                        action="store_false")
    # NTU
    parser.add_argument("--ntu-version", type=int, default=120, choices=[60, 120])
    # Charades
    parser.add_argument("--charades-csv-dir", default=None)
    # manifest
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--action-names", default=None)
    parser.add_argument("--train-identity-ratio", type=float, default=0.5,
                        help="only used by --dataset manifest without a 'split' column")
    return parser.parse_args()


if __name__ == "__main__":
    main()
