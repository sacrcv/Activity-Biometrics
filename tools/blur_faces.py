#!/usr/bin/env python
"""Gaussian-blur faces to make a dataset face-restricted.

The paper, Section 4: "To facilitate face restricted person identification the
faces are blurred using Gaussian blur for both the test and train split of all
datasets." Section 3 gives the motivation: "We are using a face restricted
setting to perform this task, where the face of the individual is blurred so as
to avoid learning any of the facial features."

This rewrites frames **in place** by default, because every consumer -- the
silhouette extractor, the dataloader, and any visualisation -- should see the
same face-restricted pixels. Pass ``--out`` to write a copy instead.

Detectors
---------
``--detector dnn`` (default)
    OpenCV's SSD ResNet-10 face detector. ~5 MB, fast enough for the ~1.8M
    frames of NTU RGB-AB, and far more reliable than Haar at the small face
    sizes and oblique angles these datasets contain. Weights are fetched once
    to ``--cache-dir``.

``--detector haar``
    Haar cascade shipped inside opencv-python. No download, noticeably weaker.

``--detector none``
    Skip detection and rely purely on the fallback below.

The fallback matters more than the detector
-------------------------------------------
A missed detection leaks exactly the cue the face-restricted setting exists to
remove, and misses are common here: most NTU subjects are back-facing, distant,
or both. So whenever no face is found in a frame, the top ``--fallback-fraction``
of the person region is blurred regardless. With ``--silhouettes-root`` that
region comes from the silhouette's bounding box; otherwise it falls back to the
top of the frame. The result is that **every** frame is blurred somewhere in the
head region, and the detector only improves how tightly.

Examples
--------
::

    # in place, with silhouettes to locate the head
    python tools/blur_faces.py --frames-root data/ntu_rgb_ab/frames \\
        --silhouettes-root data/ntu_rgb_ab/silhouettes_raw

    # write a copy, more aggressive blur
    python tools/blur_faces.py --frames-root data/ntu_rgb_ab/frames \\
        --out data/ntu_rgb_ab/frames_blurred --sigma-scale 0.2

Run this **before** ``tools/extract_silhouettes.py`` if you want silhouettes of
the blurred frames, or after if you want a clean person mask to drive the
fallback. Blurring does not change the person's outline, so either order gives
equivalent silhouettes; the second is slightly more accurate.
"""

import argparse
import os
import sys
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")

DNN_PROTO_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/dnn/face_detector/deploy.prototxt"
)
DNN_WEIGHTS_URL = (
    "https://raw.githubusercontent.com/opencv/opencv_3rdparty/"
    "dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
)


def download(url: str, path: str) -> str:
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    print(f"downloading {os.path.basename(path)} ...")
    urllib.request.urlretrieve(url, path)
    return path


class DnnFaceDetector:
    """OpenCV SSD ResNet-10 face detector."""

    def __init__(self, cache_dir: str, confidence: float = 0.5):
        import cv2

        self.cv2 = cv2
        proto = download(DNN_PROTO_URL, os.path.join(cache_dir, "deploy.prototxt"))
        weights = download(
            DNN_WEIGHTS_URL,
            os.path.join(cache_dir, "res10_300x300_ssd_iter_140000.caffemodel"),
        )
        self.net = cv2.dnn.readNetFromCaffe(proto, weights)
        self.confidence = confidence

    def __call__(self, image: np.ndarray):
        """``[H, W, 3]`` BGR -> list of ``(x1, y1, x2, y2)``."""
        cv2 = self.cv2
        height, width = image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
        )
        self.net.setInput(blob)
        detections = self.net.forward()

        boxes = []
        for i in range(detections.shape[2]):
            score = float(detections[0, 0, i, 2])
            if score < self.confidence:
                continue
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            x1, y1, x2, y2 = box.astype(int)
            boxes.append((max(x1, 0), max(y1, 0), min(x2, width), min(y2, height)))
        return boxes


class HaarFaceDetector:
    """Haar cascade bundled with opencv-python."""

    def __init__(self):
        import cv2

        self.cv2 = cv2
        path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        self.cascade = cv2.CascadeClassifier(path)
        if self.cascade.empty():
            raise RuntimeError(f"could not load the Haar cascade at {path}")

    def __call__(self, image: np.ndarray):
        gray = self.cv2.cvtColor(image, self.cv2.COLOR_BGR2GRAY)
        found = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        return [(x, y, x + w, y + h) for (x, y, w, h) in found]


class NullDetector:
    def __call__(self, image: np.ndarray):
        return []


def blur_region(image: np.ndarray, box, sigma_scale: float, expand: float) -> np.ndarray:
    """Gaussian-blur one box in place, with sigma proportional to its width."""
    import cv2

    height, width = image.shape[:2]
    x1, y1, x2, y2 = box
    box_w, box_h = x2 - x1, y2 - y1
    if box_w <= 1 or box_h <= 1:
        return image

    # Expand so the blur covers hairline and jaw, not just the detected face.
    pad_x = int(box_w * (expand - 1.0) / 2.0)
    pad_y = int(box_h * (expand - 1.0) / 2.0)
    x1, y1 = max(x1 - pad_x, 0), max(y1 - pad_y, 0)
    x2, y2 = min(x2 + pad_x, width), min(y2 + pad_y, height)
    if x2 <= x1 or y2 <= y1:
        return image

    # Sigma scales with face size so small and large faces are equally
    # unrecognisable; a fixed sigma would leave big faces readable.
    sigma = max(sigma_scale * (x2 - x1), 1.0)
    kernel = int(sigma * 6) | 1  # odd, ~3 sigma each side
    patch = image[y1:y2, x1:x2]
    image[y1:y2, x1:x2] = cv2.GaussianBlur(patch, (kernel, kernel), sigma)
    return image


def head_region_from_silhouette(mask: np.ndarray, fraction: float):
    """Bounding box of the top ``fraction`` of a silhouette, or None."""
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return None
    top, bottom = int(rows[0]), int(rows[-1])
    height = bottom - top + 1
    head_bottom = top + max(int(round(height * fraction)), 1)
    # Narrow to the columns actually occupied in the head band, so a raised arm
    # lower down does not widen the blur.
    band = mask[top:head_bottom]
    band_cols = np.where(band.any(axis=0))[0]
    if band_cols.size == 0:
        band_cols = cols
    return (int(band_cols[0]), top, int(band_cols[-1]) + 1, head_bottom)


def frame_files(directory: str):
    return sorted(n for n in os.listdir(directory) if n.lower().endswith(IMAGE_SUFFIXES))


def build_detector(args):
    if args.detector == "dnn":
        return DnnFaceDetector(args.cache_dir, args.confidence)
    if args.detector == "haar":
        return HaarFaceDetector()
    return NullDetector()


def main() -> int:
    args = parse_args()
    import cv2
    from tqdm import tqdm

    if not os.path.isdir(args.frames_root):
        raise SystemExit(f"--frames-root not found: {args.frames_root}")

    detector = build_detector(args)
    in_place = args.out is None
    print(f"detector={args.detector}, {'in place' if in_place else f'copy -> {args.out}'}")
    print(
        f"fallback: top {args.fallback_fraction:.0%} of the "
        f"{'silhouette bbox' if args.silhouettes_root else 'frame'}"
    )

    clips = sorted(
        n for n in os.listdir(args.frames_root)
        if os.path.isdir(os.path.join(args.frames_root, n))
    )
    if args.num_shards > 1:
        clips = clips[args.shard :: args.num_shards]
    if args.limit:
        clips = clips[: args.limit]

    detected = fallback = total = 0
    for clip in tqdm(clips, desc="clips"):
        source_dir = os.path.join(args.frames_root, clip)
        target_dir = source_dir if in_place else os.path.join(args.out, clip)
        os.makedirs(target_dir, exist_ok=True)

        silhouette_dir = (
            os.path.join(args.silhouettes_root, clip) if args.silhouettes_root else None
        )

        for name in frame_files(source_dir):
            path = os.path.join(source_dir, name)
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            if image is None:
                continue
            total += 1

            boxes = detector(image)
            if boxes:
                detected += 1
                for box in boxes:
                    image = blur_region(image, box, args.sigma_scale, args.expand)
            else:
                box = None
                if silhouette_dir:
                    stem = os.path.splitext(name)[0]
                    for suffix in (".png", ".jpg"):
                        candidate = os.path.join(silhouette_dir, stem + suffix)
                        if os.path.exists(candidate):
                            mask = cv2.imread(candidate, cv2.IMREAD_GRAYSCALE)
                            if mask is not None:
                                if mask.shape != image.shape[:2]:
                                    mask = cv2.resize(
                                        mask, (image.shape[1], image.shape[0]),
                                        interpolation=cv2.INTER_NEAREST,
                                    )
                                box = head_region_from_silhouette(
                                    mask > 127, args.fallback_fraction
                                )
                            break
                if box is None:
                    # Last resort: the top band of the frame. Crude, but it
                    # guarantees no frame leaves here with an untouched head.
                    height, width = image.shape[:2]
                    box = (0, 0, width, max(int(height * args.fallback_fraction), 2))
                fallback += 1
                image = blur_region(image, box, args.sigma_scale, 1.0)

            cv2.imwrite(
                os.path.join(target_dir, name),
                image,
                [int(cv2.IMWRITE_JPEG_QUALITY), args.quality],
            )

    print(f"\nblurred {total} frame(s) across {len(clips)} clip(s)")
    if total:
        print(f"  {detected} ({detected / total:.1%}) by detection, {fallback} by fallback")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--frames-root", required=True)
    parser.add_argument("--out", default=None, help="write a copy here instead of in place")
    parser.add_argument(
        "--silhouettes-root", default=None,
        help="full-resolution person masks, used to locate the head when no face is detected",
    )
    parser.add_argument("--detector", default="dnn", choices=["dnn", "haar", "none"])
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument(
        "--sigma-scale", type=float, default=0.1,
        help="Gaussian sigma as a fraction of the blurred box's width",
    )
    parser.add_argument(
        "--expand", type=float, default=1.3,
        help="expand each detected face box by this factor before blurring",
    )
    parser.add_argument(
        "--fallback-fraction", type=float, default=0.22,
        help="fraction of the person's height treated as the head region",
    )
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--cache-dir", default=os.path.expanduser("~/.cache/abnet/face_detector"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main())
