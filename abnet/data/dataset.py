"""Video dataset for activity-biometrics.

Both benchmarks (NTU RGB-AB and Charades-AB) are consumed through one
annotation schema, produced by ``tools/prepare_dataset.py``::

    {
      "name": "ntu_rgb_ab",
      "num_actions": 94,
      "action_names": ["drink water", ...],
      "multilabel": false,
      "splits": {
        "train":   [<sample>, ...],
        "query":   [<sample>, ...],
        "gallery": [<sample>, ...]
      }
    }

where each ``<sample>`` is::

    {
      "id":              "S001C001P001R001A001",   # unique clip id
      "frames_dir":      "ntu_rgb_ab/frames/S001C001P001R001A001",
      "silhouettes_dir": "ntu_rgb_ab/silhouettes/S001C001P001R001A001",
      "num_frames":      103,
      "video":           null,     # alternative to frames_dir
      "pid":             12,       # identity label
      "action":          7,        # int, or list[int] when multilabel
      "view":            2         # camera / viewpoint id
    }

Paths are relative to ``data.data_root``. ``silhouettes_dir`` is optional and
train-only: it feeds the bias-less teacher (Section 3.1), and the paper is
explicit that "we only use silhouette during training and it is not required
for inference".

Identity labels are remapped to a contiguous ``[0, num_train_identities)``
range for the train split (the classifier needs that); query and gallery keep
the original ids so retrieval can match across splits.
"""

import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from abnet.data.transforms import ClipTransform
from abnet.utils.logger import get_logger


def load_annotation(path: str) -> Dict:
    with open(path) as f:
        return json.load(f)


def primary_action(action) -> int:
    """Single action id for a sample; ``-1`` when unlabelled.

    Multi-label samples (Charades-AB) collapse to their first label, which is
    only ever used for the cross-activity gallery filter, never for the loss.
    """
    if action is None:
        return -1
    if isinstance(action, (list, tuple)):
        return int(action[0]) if len(action) else -1
    return int(action)


def sample_frame_indices(
    num_frames: int,
    num_sampled: int = 8,
    stride: int = 4,
    mode: str = "train",
    sampling: str = "stride",
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Pick ``num_sampled`` frame indices from a clip of ``num_frames`` frames.

    ``sampling="stride"`` implements the paper's recipe -- "We create RGB video
    clips from each original video by randomly selecting 8 frames with a stride
    of 4" -- as a contiguous strided window whose offset is random during
    training and centred during evaluation.

    ``sampling="linspace"`` spreads the frames over the whole clip instead
    (random within each of ``num_sampled`` equal segments when training), which
    is the better choice for the long untrimmed videos of Charades-AB.
    """
    num_frames = max(int(num_frames), 1)
    rng = rng or np.random.default_rng()

    if sampling == "linspace":
        bounds = np.linspace(0, num_frames, num_sampled + 1)
        if mode == "train":
            lows, highs = bounds[:-1], bounds[1:]
            indices = lows + rng.random(num_sampled) * np.maximum(highs - lows, 1e-6)
            indices = np.floor(indices)
        else:
            indices = np.rint((bounds[:-1] + bounds[1:]) / 2.0 - 0.5)
        return np.clip(indices, 0, num_frames - 1).astype(np.int64)

    if sampling != "stride":
        raise ValueError(f"unknown sampling '{sampling}'; use 'stride' or 'linspace'")

    if num_sampled > 1 and num_frames < (num_sampled - 1) * stride + 1:
        # Clip too short for the requested stride. Shrink the stride to the
        # largest value that still fits rather than clamping, which would
        # otherwise return many copies of the final frame.
        stride = max((num_frames - 1) // (num_sampled - 1), 1)

    span = (num_sampled - 1) * stride + 1
    slack = num_frames - span
    if slack > 0:
        start = int(rng.integers(0, slack + 1)) if mode == "train" else slack // 2
    else:
        start = 0
    indices = start + np.arange(num_sampled) * stride
    return np.clip(indices, 0, num_frames - 1).astype(np.int64)


class VideoReIDDataset(Dataset):
    """One split of an activity-biometrics benchmark."""

    def __init__(
        self,
        samples: List[Dict],
        data_root: str,
        transform: ClipTransform,
        num_sampled_frames: int = 8,
        frame_stride: int = 4,
        sampling: str = "stride",
        mode: str = "train",
        frame_template: str = "frame{:06d}.jpg",
        silhouette_template: str = "frame{:06d}.png",
        frame_index_base: int = 1,
        num_actions: int = 1,
        multilabel: bool = False,
        pid_map: Optional[Dict[int, int]] = None,
        load_silhouettes: bool = False,
        make_distorted: bool = False,
    ):
        self.samples = samples
        self.data_root = data_root
        self.transform = transform
        self.num_sampled_frames = num_sampled_frames
        self.frame_stride = frame_stride
        self.sampling = sampling
        self.mode = mode
        self.frame_template = frame_template
        self.silhouette_template = silhouette_template
        self.frame_index_base = frame_index_base
        self.num_actions = num_actions
        self.multilabel = multilabel
        self.pid_map = pid_map
        self.load_silhouettes = load_silhouettes
        self.make_distorted = make_distorted

    def __len__(self) -> int:
        return len(self.samples)

    # ------------------------------------------------------------------
    # labels
    # ------------------------------------------------------------------
    @property
    def pids(self) -> np.ndarray:
        return np.array([s["pid"] for s in self.samples], dtype=np.int64)

    @property
    def views(self) -> np.ndarray:
        return np.array([s.get("view", 0) for s in self.samples], dtype=np.int64)

    @property
    def actions(self) -> List:
        return [s.get("action", -1) for s in self.samples]

    def mapped_pid(self, raw_pid: int) -> int:
        if self.pid_map is None:
            return int(raw_pid)
        return int(self.pid_map.get(int(raw_pid), -1))

    def _action_target(self, action):
        if self.multilabel:
            target = torch.zeros(self.num_actions, dtype=torch.float32)
            if action is None:
                return target
            if isinstance(action, (list, tuple, set)):
                for a in action:
                    if 0 <= int(a) < self.num_actions:
                        target[int(a)] = 1.0
            elif 0 <= int(action) < self.num_actions:
                target[int(action)] = 1.0
            return target
        if action is None:
            return torch.tensor(-1, dtype=torch.long)
        if isinstance(action, (list, tuple)):
            action = action[0] if action else -1
        return torch.tensor(int(action), dtype=torch.long)

    # ------------------------------------------------------------------
    # frame / silhouette loading
    # ------------------------------------------------------------------
    def _resolve(self, relative: str) -> str:
        if os.path.isabs(relative):
            return relative
        return os.path.join(self.data_root, relative)

    def _frame_path(self, sample: Dict, frames_dir: str, index: int, template: str) -> str:
        """Resolve one frame of a sample to a file path.

        Three layouts are supported:

        * ``frame_names`` -- an explicit per-frame file list, for sources whose
          frames are named arbitrarily;
        * ``frame_offset`` -- the clip is a segment of a longer video whose
          frames all live in one directory, as when action segments are cut out
          of an untrimmed recording;
        * otherwise ``template`` formatted with the frame number.
        """
        names = sample.get("frame_names")
        if names:
            name = names[min(int(index), len(names) - 1)]
            return name if os.path.isabs(name) else os.path.join(frames_dir, os.path.basename(name))

        offset = int(sample.get("frame_offset", 0))
        return os.path.join(
            frames_dir, template.format(int(index) + offset + self.frame_index_base)
        )

    def _load_images(
        self, sample: Dict, directory: str, indices: np.ndarray, template: str, mode: str
    ) -> torch.Tensor:
        from PIL import Image

        root = self._resolve(directory)
        images = []
        for i in indices:
            path = self._frame_path(sample, root, int(i), template)
            with Image.open(path) as img:
                images.append(torch.from_numpy(np.array(img.convert(mode))))
        stacked = torch.stack(images)
        if stacked.ndim == 3:  # grayscale: [T, H, W] -> [T, 1, H, W]
            stacked = stacked.unsqueeze(-1)
        clip = stacked.permute(0, 3, 1, 2)  # [T, C, H, W], uint8
        return clip.float() / 255.0

    def _load_frames_from_video(self, sample: Dict, indices: np.ndarray) -> torch.Tensor:
        import decord

        path = self._resolve(sample["video"])
        reader = decord.VideoReader(path, num_threads=1)
        offset = int(sample.get("frame_offset", 0))
        idx = np.clip(indices + offset, 0, len(reader) - 1)
        batch = reader.get_batch(idx.tolist()).asnumpy()  # [T, H, W, 3]
        clip = torch.from_numpy(batch).permute(0, 3, 1, 2)
        return clip.float() / 255.0

    def _num_frames(self, sample: Dict) -> int:
        if sample.get("num_frames"):
            return int(sample["num_frames"])
        if sample.get("frame_names"):
            return len(sample["frame_names"])
        if sample.get("frames_dir"):
            frames_dir = self._resolve(sample["frames_dir"])
            count = len([f for f in os.listdir(frames_dir) if not f.startswith(".")])
            sample["num_frames"] = count
            return count
        if sample.get("video"):
            import decord

            count = len(decord.VideoReader(self._resolve(sample["video"]), num_threads=1))
            sample["num_frames"] = count
            return count
        raise ValueError(f"sample {sample.get('id')} has neither frames_dir nor video")

    # ------------------------------------------------------------------
    def __getitem__(self, index: int) -> Dict:
        sample = self.samples[index]
        total_frames = self._num_frames(sample)
        indices = sample_frame_indices(
            total_frames,
            num_sampled=self.num_sampled_frames,
            stride=self.frame_stride,
            mode=self.mode,
            sampling=self.sampling,
        )

        if sample.get("frames_dir"):
            clip = self._load_images(
                sample, sample["frames_dir"], indices, self.frame_template, "RGB"
            )
        elif sample.get("video"):
            clip = self._load_frames_from_video(sample, indices)
        else:
            raise ValueError(f"sample {sample['id']} has neither frames_dir nor video")

        # The teacher consumes exactly the frames the student sees, so the
        # per-sample KL of Eq. 1 compares two views of the same clip.
        silhouette = None
        if self.load_silhouettes and sample.get("silhouettes_dir"):
            silhouette = self._load_images(
                sample, sample["silhouettes_dir"], indices, self.silhouette_template, "L"
            )

        transformed = self.transform(
            clip,
            silhouette=silhouette,
            clip_id=str(sample["id"]),
            make_distorted=self.make_distorted,
        )

        item = {
            "index": index,
            "id": sample["id"],
            "pid": torch.tensor(self.mapped_pid(sample["pid"]), dtype=torch.long),
            "raw_pid": torch.tensor(int(sample["pid"]), dtype=torch.long),
            "view": torch.tensor(int(sample.get("view", 0)), dtype=torch.long),
            "action": self._action_target(sample.get("action")),
            "action_id": torch.tensor(primary_action(sample.get("action")), dtype=torch.long),
            "has_silhouette": torch.tensor(
                1.0 if "silhouette" in transformed else 0.0, dtype=torch.float32
            ),
        }
        item.update(transformed)
        if self.load_silhouettes and "silhouette" not in item:
            # Keep the batch shape uniform when only some clips have masks, so
            # collate still works; the teacher loss masks these rows out.
            height, width = self.transform.silhouette_size
            item["silhouette"] = torch.zeros(
                self.num_sampled_frames, 1, height, width, dtype=torch.float32
            )
        return item


def collate_video_batch(batch: Sequence[Dict]) -> Dict:
    """Stack tensor fields, keep ids as a list of strings."""
    out: Dict = {}
    tensor_keys = [k for k, v in batch[0].items() if isinstance(v, torch.Tensor)]
    for key in tensor_keys:
        out[key] = torch.stack([item[key] for item in batch])

    out["id"] = [item["id"] for item in batch]
    out["index"] = torch.tensor([item["index"] for item in batch], dtype=torch.long)
    return out


def build_pid_map(samples: Sequence[Dict]) -> Dict[int, int]:
    """Map the raw identity labels of a split onto ``[0, num_identities)``."""
    unique = sorted({int(s["pid"]) for s in samples})
    return {pid: i for i, pid in enumerate(unique)}


def describe_split(name: str, samples: Sequence[Dict]) -> str:
    pids = {int(s["pid"]) for s in samples}
    views = {int(s.get("view", 0)) for s in samples}
    with_sil = sum(1 for s in samples if s.get("silhouettes_dir"))
    return (
        f"{name:>8}: {len(samples):6d} clips | {len(pids):4d} identities | "
        f"{len(views):3d} views | {with_sil:6d} with silhouettes"
    )


def log_dataset_summary(annotation: Dict) -> None:
    logger = get_logger()
    logger.info(
        f"dataset '{annotation.get('name', '?')}' "
        f"({annotation.get('num_actions')} actions, "
        f"multilabel={annotation.get('multilabel', False)}, "
        f"protocol={annotation.get('protocol', '?')})"
    )
    for split, samples in annotation["splits"].items():
        logger.info("  " + describe_split(split, samples))
