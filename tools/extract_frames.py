#!/usr/bin/env python
"""Decode videos into JPEG frame directories.

Reading pre-extracted frames is much faster than seeking in compressed video
during training, which is why the annotation schema prefers ``frames_dir``.

    python tools/extract_frames.py --videos data/charades_ab/videos \
        --out data/charades_ab/frames --fps 10

Produces ``<out>/<video_stem>/frame000001.jpg``, matching the default
``data.frame_template``.
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

VIDEO_SUFFIXES = (".mp4", ".avi", ".mkv", ".mov", ".webm", ".m4v")


def extract_one(video_path, out_dir, fps, max_frames, short_side, quality, overwrite):
    import numpy as np
    from PIL import Image

    if os.path.isdir(out_dir) and not overwrite:
        existing = [f for f in os.listdir(out_dir) if f.endswith(".jpg")]
        if existing:
            return out_dir, len(existing), "skipped"

    os.makedirs(out_dir, exist_ok=True)
    try:
        import decord

        reader = decord.VideoReader(video_path, num_threads=1)
        native_fps = float(reader.get_avg_fps()) or 30.0
        total = len(reader)

        if fps and fps > 0:
            step = max(native_fps / fps, 1.0)
            indices = np.arange(0, total, step).astype(int)
        else:
            indices = np.arange(total)
        if max_frames and len(indices) > max_frames:
            indices = np.linspace(0, len(indices) - 1, max_frames).astype(int)
            indices = np.arange(0, total, max(native_fps / (fps or native_fps), 1.0)).astype(int)[indices]

        written = 0
        # Decode in blocks to bound memory on long videos.
        block = 256
        for offset in range(0, len(indices), block):
            batch = indices[offset : offset + block]
            frames = reader.get_batch(batch.tolist()).asnumpy()
            for i, frame in enumerate(frames):
                image = Image.fromarray(frame)
                if short_side and min(image.size) > short_side:
                    scale = short_side / min(image.size)
                    image = image.resize(
                        (round(image.width * scale), round(image.height * scale)),
                        Image.BICUBIC,
                    )
                image.save(
                    os.path.join(out_dir, f"frame{offset + i + 1:06d}.jpg"), quality=quality
                )
                written += 1
        return out_dir, written, "ok"
    except Exception as error:  # noqa: BLE001
        return out_dir, 0, f"failed: {error}"


def main():
    args = parse_args()

    videos = []
    for root, _, files in os.walk(args.videos):
        for name in sorted(files):
            if name.lower().endswith(VIDEO_SUFFIXES):
                videos.append(os.path.join(root, name))
    print(f"found {len(videos)} videos under {args.videos}")

    jobs = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for video in videos:
            stem = os.path.splitext(os.path.basename(video))[0]
            jobs.append(
                pool.submit(
                    extract_one, video, os.path.join(args.out, stem), args.fps,
                    args.max_frames, args.short_side, args.quality, args.overwrite,
                )
            )

        counts = {"ok": 0, "skipped": 0, "failed": 0}
        for future in tqdm(as_completed(jobs), total=len(jobs), desc="extract"):
            out_dir, written, status = future.result()
            if status.startswith("failed"):
                counts["failed"] += 1
                print(f"  {os.path.basename(out_dir)}: {status}")
            else:
                counts[status] += 1

    print(f"done: {counts['ok']} extracted, {counts['skipped']} skipped, "
          f"{counts['failed']} failed -> {args.out}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--videos", required=True, help="directory of video files")
    parser.add_argument("--out", required=True, help="output frames root")
    parser.add_argument("--fps", type=float, default=10.0,
                        help="target sampling rate; 0 keeps every frame")
    parser.add_argument("--max-frames", type=int, default=0, help="0 = unlimited")
    parser.add_argument("--short-side", type=int, default=256,
                        help="downscale so the short side is this many pixels; 0 disables")
    parser.add_argument("--quality", type=int, default=90)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
