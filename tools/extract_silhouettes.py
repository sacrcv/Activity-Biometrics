#!/usr/bin/env python
"""Extract binary person silhouettes from RGB frames with Mask2Former.

Step 1 of the three-stage pipeline. The paper: "The silhouettes of the RGB
videos are extracted using Mask2Former [8] to use as input to T_theta(.)".

What this does per frame
------------------------
1. Run Mask2Former COCO **instance** segmentation.
2. Keep the highest-scoring ``person`` instance. One subject per clip is the
   right assumption here: NTU RGB-AB excludes the two-person "mutual" classes,
   and Charades-AB is single-actor.
3. Binarise the mask at 0.5.

Then per clip, GaitGL's pretreatment (``pretreatment_oumvlp.py`` upstream):
crop to the mask's bounding box, scale to the target height, and centre the
subject horizontally **on the silhouette's centre of mass** rather than on the
bbox centre. That distinction matters: centre of mass is stable when an arm or
leg extends to one side, a bbox centre is not, and GaitGL's learned part
partitioning assumes the body sits in a consistent place.

Output is ``<out>/<clip_id>/frame%06d.png``, 8-bit single channel, 0 or 255, at
64x44 -- GaitGL's native input size.

Backends
--------
``--backend hf`` (default)
    HuggingFace ``transformers``. Same Facebook weights as the official
    release, installable with pip, no detectron2 build. This is the path that
    gets exercised by the test suite.

``--backend d2``
    The official `facebookresearch/Mask2Former
    <https://github.com/facebookresearch/Mask2Former>`_ repo via detectron2,
    for bit-exactness with the paper. Needs ``--d2-config`` and
    ``--d2-weights``, and detectron2 + the Mask2Former repo on ``PYTHONPATH``.

Appendix C notes that a stronger segmenter helps ("with Grounded-SAM as
silhouette extractor the performance does go up"), and that the choice is not
critical because extraction happens offline and never at inference.

Examples
--------
::

    python tools/extract_silhouettes.py \\
        --frames-root data/ntu_rgb_ab/frames \\
        --out data/ntu_rgb_ab/silhouettes

    # a specific model, larger batches, half precision
    python tools/extract_silhouettes.py \\
        --frames-root data/ntu_rgb_ab/frames \\
        --out data/ntu_rgb_ab/silhouettes \\
        --model facebook/mask2former-swin-large-coco-instance \\
        --batch-size 16 --dtype fp16

    # shard across 8 GPUs: run this 8 times, one per GPU
    for i in $(seq 0 7); do
      CUDA_VISIBLE_DEVICES=$i python tools/extract_silhouettes.py \\
        --frames-root data/ntu_rgb_ab/frames --out data/ntu_rgb_ab/silhouettes \\
        --shard $i --num-shards 8 &
    done; wait
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")

#: COCO instance id for "person" in the Mask2Former COCO label space.
COCO_PERSON_LABEL = 0

#: GaitGL's input size, (height, width).
TARGET_SIZE = (64, 44)


# ----------------------------------------------------------------------
# segmentation backends
# ----------------------------------------------------------------------
class HFMask2Former:
    """Mask2Former through ``transformers``."""

    def __init__(self, model_name: str, device: str = "cuda", dtype: str = "fp32"):
        import torch
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

        self.torch = torch
        self.device = device
        self.dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            model_name, torch_dtype=self.dtype
        )
        self.model.to(device).eval()

    def __call__(self, images):
        """``list[PIL.Image] -> list[np.ndarray[bool]]`` at each image's size."""
        torch = self.torch
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {
            k: (v.to(self.device, dtype=self.dtype) if v.is_floating_point() else v.to(self.device))
            for k, v in inputs.items()
        }
        with torch.no_grad():
            outputs = self.model(**inputs)

        sizes = [(img.height, img.width) for img in images]
        processed = self.processor.post_process_instance_segmentation(
            outputs, target_sizes=sizes, return_binary_maps=True
        )

        masks = []
        for result, (height, width) in zip(processed, sizes):
            masks.append(_pick_person(result, height, width))
        return masks


def _pick_person(result, height: int, width: int) -> np.ndarray:
    """Highest-scoring ``person`` instance from one post-processed result."""
    segmentation = result.get("segmentation")
    info = result.get("segments_info") or []
    if segmentation is None or len(info) == 0:
        return np.zeros((height, width), dtype=bool)

    best, best_score = None, -1.0
    for position, segment in enumerate(info):
        if int(segment.get("label_id", -1)) != COCO_PERSON_LABEL:
            continue
        score = float(segment.get("score", 1.0))
        if score > best_score:
            best, best_score = position, score

    if best is None:
        return np.zeros((height, width), dtype=bool)

    seg = segmentation.cpu().numpy() if hasattr(segmentation, "cpu") else np.asarray(segmentation)
    if seg.ndim == 3:
        # return_binary_maps=True gives one binary map per segment.
        return seg[best] > 0.5
    # Otherwise it is an id map; segments_info[i] corresponds to id i.
    return seg == int(info[best].get("id", best))


class D2Mask2Former:
    """Mask2Former through the official repo + detectron2."""

    def __init__(self, config_path: str, weights_path: str, device: str = "cuda",
                 score_threshold: float = 0.5):
        try:
            from detectron2.config import get_cfg
            from detectron2.engine import DefaultPredictor
            from detectron2.projects.deeplab import add_deeplab_config
            from mask2former import add_maskformer2_config
        except ImportError as exc:  # pragma: no cover - optional path
            raise ImportError(
                "--backend d2 needs detectron2 and the Mask2Former repo importable. "
                "Install per https://github.com/facebookresearch/Mask2Former/blob/main/INSTALL.md "
                "or use the default --backend hf, which loads the same weights."
            ) from exc

        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        cfg.merge_from_file(config_path)
        cfg.MODEL.WEIGHTS = weights_path
        cfg.MODEL.DEVICE = device
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = score_threshold
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold
        cfg.freeze()
        self.predictor = DefaultPredictor(cfg)

    def __call__(self, images):
        masks = []
        for image in images:
            # detectron2 expects BGR uint8.
            array = np.array(image.convert("RGB"))[:, :, ::-1]
            instances = self.predictor(array)["instances"].to("cpu")
            keep = instances.pred_classes == COCO_PERSON_LABEL
            if keep.sum() == 0:
                masks.append(np.zeros(array.shape[:2], dtype=bool))
                continue
            scores = instances.scores[keep]
            best = int(scores.argmax())
            masks.append(instances.pred_masks[keep][best].numpy() > 0.5)
        return masks


# ----------------------------------------------------------------------
# GaitGL pretreatment
# ----------------------------------------------------------------------
def align_silhouette(mask: np.ndarray, size=TARGET_SIZE) -> np.ndarray:
    """Crop, scale and horizontally centre one mask, GaitGL-style.

    Returns a ``uint8`` array of ``size`` with values in ``{0, 255}``. An empty
    mask yields an all-zero frame rather than an error: a few missed detections
    in a clip are normal, and the teacher loss tolerates them.
    """
    from PIL import Image

    height, width = size
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return np.zeros(size, dtype=np.uint8)

    cropped = mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]

    # Scale by height, preserving aspect: gait cues live in body proportions,
    # so stretching to a fixed width would destroy them.
    crop_h, crop_w = cropped.shape
    scaled_w = max(int(round(crop_w * height / crop_h)), 1)
    image = Image.fromarray((cropped * 255).astype(np.uint8)).resize(
        (scaled_w, height), Image.BILINEAR
    )
    scaled = np.array(image)

    # Horizontal centring on the centre of mass (see the module docstring).
    column_mass = scaled.astype(np.float64).sum(axis=0)
    total = column_mass.sum()
    if total <= 0:
        return np.zeros(size, dtype=np.uint8)
    centre = int(round(float((np.arange(scaled_w) * column_mass).sum() / total)))

    half = width // 2
    left, right = centre - half, centre - half + width
    canvas = np.zeros((height, width), dtype=np.uint8)
    source_left, source_right = max(left, 0), min(right, scaled_w)
    if source_right > source_left:
        canvas[:, source_left - left : source_right - left] = scaled[
            :, source_left:source_right
        ]
    return (canvas > 127).astype(np.uint8) * 255


# ----------------------------------------------------------------------
def list_clips(frames_root: str, shard: int, num_shards: int):
    names = sorted(
        name for name in os.listdir(frames_root)
        if os.path.isdir(os.path.join(frames_root, name))
    )
    if num_shards > 1:
        names = names[shard::num_shards]
    return names


def frame_files(directory: str):
    return sorted(
        name for name in os.listdir(directory) if name.lower().endswith(IMAGE_SUFFIXES)
    )


def build_backend(args):
    if args.backend == "hf":
        return HFMask2Former(args.model, device=args.device, dtype=args.dtype)
    if args.backend == "d2":
        if not (args.d2_config and args.d2_weights):
            raise SystemExit("--backend d2 requires --d2-config and --d2-weights")
        return D2Mask2Former(args.d2_config, args.d2_weights, device=args.device)
    raise SystemExit(f"unknown backend '{args.backend}'")


def main() -> int:
    args = parse_args()
    from PIL import Image
    from tqdm import tqdm

    if not os.path.isdir(args.frames_root):
        raise SystemExit(f"--frames-root not found: {args.frames_root}")

    clips = list_clips(args.frames_root, args.shard, args.num_shards)
    if args.limit:
        clips = clips[: args.limit]
    if not clips:
        raise SystemExit(f"no clip directories under {args.frames_root}")

    print(f"{len(clips)} clip(s) to process (shard {args.shard}/{args.num_shards})")
    print(f"backend={args.backend}, target size={TARGET_SIZE[0]}x{TARGET_SIZE[1]}")

    segmenter = build_backend(args)

    done = skipped = empty_frames = 0
    for clip in tqdm(clips, desc="clips"):
        source_dir = os.path.join(args.frames_root, clip)
        target_dir = os.path.join(args.out, clip)
        names = frame_files(source_dir)
        if not names:
            continue

        if (
            not args.overwrite
            and os.path.isdir(target_dir)
            and len(frame_files(target_dir)) >= len(names)
        ):
            skipped += 1
            continue

        os.makedirs(target_dir, exist_ok=True)
        for start in range(0, len(names), args.batch_size):
            chunk = names[start : start + args.batch_size]
            images = [Image.open(os.path.join(source_dir, n)).convert("RGB") for n in chunk]
            masks = segmenter(images)
            for name, mask in zip(chunk, masks):
                aligned = align_silhouette(mask, TARGET_SIZE)
                if not aligned.any():
                    empty_frames += 1
                stem = os.path.splitext(name)[0]
                Image.fromarray(aligned).save(os.path.join(target_dir, f"{stem}.png"))
            for image in images:
                image.close()
        done += 1

    print(f"\nwrote silhouettes for {done} clip(s) -> {args.out}")
    if skipped:
        print(f"  skipped {skipped} already-complete clip(s); --overwrite to redo")
    if empty_frames:
        print(
            f"  {empty_frames} frame(s) had no detected person and are all-zero. "
            f"A small number is expected (occlusion, subject out of frame); a large "
            f"fraction suggests the wrong --model or unusually small subjects."
        )
    print("\nNext, rebuild the annotation so the clips carry 'silhouettes_dir':")
    print(f"  python tools/prepare_dataset.py --dataset <ntu|charades> ... \\")
    print(f"      --silhouettes-root {args.out}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--frames-root", required=True, help="directory of per-clip frame dirs")
    parser.add_argument("--out", required=True, help="where to write per-clip silhouette dirs")
    parser.add_argument("--backend", default="hf", choices=["hf", "d2"])
    parser.add_argument(
        "--model",
        default="facebook/mask2former-swin-large-coco-instance",
        help="HuggingFace model id for --backend hf",
    )
    parser.add_argument("--d2-config", default=None, help="Mask2Former yaml for --backend d2")
    parser.add_argument("--d2-weights", default=None, help=".pkl weights for --backend d2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="process only the first N clips")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main())
