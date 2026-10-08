"""Clip-level transforms for ABNet.

Every stochastic decision is made **once per clip**, never per frame, so the
temporal structure a 3D CNN relies on is preserved.

Three things here are specific to ABNet and worth reading before changing:

**Biometrics distortion (Section 3.1, "Bias learning").** The distortion branch
needs a clip whose *identity* is destroyed but whose *appearance* is untouched,
because :math:`\\mathcal{L}_{Dis}` (Eq. 5) treats the appearance features of the
two clips as a positive pair and the biometrics features as a hard negative
pair. So the distorted clip is derived from the original **after** resizing,
hue shifting and flipping, and the only difference between them is one elastic
displacement field. Getting this order wrong (e.g. drawing an independent flip
for the distorted clip) silently breaks the positive pair and the loss stops
meaning anything.

**Hue shifting (Section 4, "Datasets").** The paper says "the videos from all
five datasets undergo an arbitrarily chosen value of hue shifting", i.e. it is a
fixed property of each video rather than a per-epoch augmentation. The shift is
therefore derived deterministically from the clip id, which reproduces that
without writing a second copy of the dataset to disk. It is applied in eval too,
for the same reason.

**Silhouettes.** The teacher is GaitGL, which expects 64x44 aligned binary
silhouettes, while the student reads 256x128 RGB. The two streams therefore have
different spatial sizes but must share the horizontal flip, or the distillation
target stops corresponding to the student's input.
"""

import hashlib
import random
from typing import Dict, Optional, Sequence, Tuple

import torch
import torchvision.transforms.functional as TF

#: Kinetics statistics, from the 3D-ResNets-PyTorch repository whose
#: Kinetics-700 ResNet-3D weights we load (``get_mean_std(1, 'kinetics')``).
KINETICS_MEAN = (0.4345, 0.4051, 0.3775)
KINETICS_STD = (0.2768, 0.2713, 0.2737)

#: Paper, "Implementation and training details": "Every input frame undergoes
#: resizing to dimensions of 256x128." Stored as (height, width).
DEFAULT_IMAGE_SIZE: Tuple[int, int] = (256, 128)

#: GaitGL's input size. ``tools/extract_silhouettes.py`` writes silhouettes at
#: this size already; resizing here is a safety net for other sources.
DEFAULT_SILHOUETTE_SIZE: Tuple[int, int] = (64, 44)


def stable_hue_factor(key: str, magnitude: float = 0.5) -> float:
    """A deterministic hue shift in ``[-magnitude, +magnitude]`` for ``key``.

    Hashing the clip id rather than drawing from an RNG means the shift is a
    stable property of the video: identical across epochs, across ranks, and
    across train/eval, which is what the paper describes.
    """
    if magnitude <= 0:
        return 0.0
    digest = hashlib.sha1(key.encode("utf-8")).digest()
    # 24 bits is ample resolution for a hue angle and keeps the maths exact.
    raw = int.from_bytes(digest[:3], "big") / float(1 << 24)
    return float((raw * 2.0 - 1.0) * magnitude)


def elastic_displacement(
    size: Sequence[int],
    alpha: float,
    sigma: float = 5.0,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """One elastic displacement field of shape ``[1, H, W, 2]``.

    Reproduces ``torchvision.transforms.ElasticTransform.get_params`` exactly,
    including its scaling quirk: the **x** displacement is divided by the
    *height* and the **y** displacement by the *width*
    (``dx *= alpha / size[0]``, ``dy *= alpha / size[1]`` with
    ``size = [H, W]``). At the paper's 256x128 input that makes the distortion
    markedly anisotropic -- about 2px of horizontal and 9px of vertical
    displacement at ``alpha=250``.

    That asymmetry is arguably accidental in torchvision, but the paper's
    Figure 4 sweep selected ``alpha=250`` *against this implementation*, so
    matching it is what makes 250 mean the same thing here. Do not "fix" it
    without re-tuning alpha.
    """
    height, width = int(size[0]), int(size[1])

    def axis(scale: float) -> torch.Tensor:
        field = torch.rand([1, 1, height, width], generator=generator) * 2 - 1
        if sigma > 0.0:
            # Blurring is what turns uniform noise into the smooth
            # "see-through-water" warp, and it also shrinks the amplitude by
            # roughly the kernel width -- which is why alpha is in the hundreds.
            kernel = int(8 * sigma + 1)
            if kernel % 2 == 0:
                kernel += 1
            field = TF.gaussian_blur(field, [kernel, kernel], [sigma, sigma])
        return field * alpha / scale

    dx = axis(float(height))
    dy = axis(float(width))
    return torch.cat([dx, dy], dim=1).permute(0, 2, 3, 1)


def apply_elastic(clip: torch.Tensor, displacement: torch.Tensor) -> torch.Tensor:
    """Warp every frame of ``[T, C, H, W]`` by the *same* displacement field.

    Two deliberate choices:

    **One field for the whole clip.** Folding time into the channel axis
    guarantees it: a per-frame field would make the distorted clip temporally
    incoherent, and the distortion branch is supposed to show a *different
    body*, not a flickering one.

    **Border padding, not zero fill.** ``torchvision.elastic_transform`` fills
    out-of-bounds samples with 0, which at ``alpha=250`` turns ~3% of a
    256x128 frame into black bands. Black bands are an *appearance* change,
    and ``f_ba``/``f_ba^D`` are a positive pair in Eq. 5 -- so zero fill would
    inject exactly the signal the loss is trying to hold constant, teaching the
    model that "distorted" means "has black edges". Replicating the border
    scrambles morphology while leaving the colour statistics alone.
    """
    num_frames, channels, height, width = clip.shape
    folded = clip.reshape(1, num_frames * channels, height, width)

    # Identity sampling grid, then add the displacement. This replicates
    # torchvision's ``_create_identity_grid``: the pixel-*centre* convention
    # ``linspace((-s+1)/s, (s-1)/s, s)`` that pairs with
    # ``align_corners=False``. A plain ``linspace(-1, 1)`` would be the
    # align_corners=True grid and would resample the image by half a pixel
    # even at alpha=0.
    base_y, base_x = torch.meshgrid(
        torch.linspace(
            (-height + 1) / height, (height - 1) / height, height,
            device=clip.device, dtype=clip.dtype,
        ),
        torch.linspace(
            (-width + 1) / width, (width - 1) / width, width,
            device=clip.device, dtype=clip.dtype,
        ),
        indexing="ij",
    )
    identity = torch.stack([base_x, base_y], dim=-1).unsqueeze(0)
    grid = identity + displacement.to(device=clip.device, dtype=clip.dtype)

    warped = torch.nn.functional.grid_sample(
        folded, grid, mode="bilinear", padding_mode="border", align_corners=False
    )
    return warped.reshape(num_frames, channels, height, width)


class ClipTransform:
    """Prepare one clip (and optionally its silhouettes and distorted twin).

    Args:
        image_size: ``(height, width)`` of the RGB stream.
        silhouette_size: ``(height, width)`` of the silhouette stream.
        train: enables the stochastic augmentations.
        horizontal_flip: probability of flipping the clip; shared with the
            silhouettes and the distorted clip.
        random_crop_pad: pad-and-random-crop amount. 0 (the default) gives the
            paper's resize-and-flip recipe.
        random_erasing: probability of erasing one rectangle, at the same
            location in every frame. Off by default: it is an appearance edit,
            and applying it to only one side of the Eq. 5 positive pair would
            corrupt the appearance supervision.
        hue_shift: magnitude of the deterministic per-video hue shift.
        distortion_alpha: elastic ``alpha``; the paper selects 250 (Figure 4).
        distortion_sigma: elastic ``sigma``.
        mean / std: channel normalisation.
    """

    def __init__(
        self,
        image_size: Sequence[int] = DEFAULT_IMAGE_SIZE,
        silhouette_size: Sequence[int] = DEFAULT_SILHOUETTE_SIZE,
        train: bool = False,
        horizontal_flip: float = 0.5,
        random_crop_pad: int = 0,
        random_erasing: float = 0.0,
        hue_shift: float = 0.5,
        distortion_alpha: float = 250.0,
        distortion_sigma: float = 5.0,
        mean: Sequence[float] = KINETICS_MEAN,
        std: Sequence[float] = KINETICS_STD,
    ):
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.silhouette_size = (int(silhouette_size[0]), int(silhouette_size[1]))
        self.train = train
        self.horizontal_flip = horizontal_flip
        self.random_crop_pad = int(random_crop_pad)
        self.random_erasing = random_erasing
        self.hue_shift = float(hue_shift)
        self.distortion_alpha = float(distortion_alpha)
        self.distortion_sigma = float(distortion_sigma)
        self.mean = list(mean)
        self.std = list(std)

    # ------------------------------------------------------------------
    @staticmethod
    def _to_unit_float(clip: torch.Tensor) -> torch.Tensor:
        if clip.dtype != torch.float32:
            clip = clip.float()
        if clip.max() > 1.5:  # arrived as uint8-valued floats
            clip = clip / 255.0
        return clip

    def _resize_rgb(self, clip: torch.Tensor) -> torch.Tensor:
        height, width = self.image_size
        if self.train and self.random_crop_pad > 0:
            pad = self.random_crop_pad
            clip = TF.resize(clip, [height + 2 * pad, width + 2 * pad], antialias=True)
            top = random.randint(0, 2 * pad)
            left = random.randint(0, 2 * pad)
            return TF.crop(clip, top, left, height, width)
        return TF.resize(clip, [height, width], antialias=True)

    def _erase(self, clip: torch.Tensor) -> torch.Tensor:
        """Erase one rectangle at the same position across all frames."""
        _, _, height, width = clip.shape
        for _ in range(10):
            area = height * width * random.uniform(0.02, 0.2)
            ratio = random.uniform(0.3, 3.3)
            erase_h = int(round((area * ratio) ** 0.5))
            erase_w = int(round((area / ratio) ** 0.5))
            if erase_h < height and erase_w < width:
                top = random.randint(0, height - erase_h)
                left = random.randint(0, width - erase_w)
                clip[:, :, top : top + erase_h, left : left + erase_w] = torch.randn(
                    clip.shape[0], clip.shape[1], erase_h, erase_w
                )
                return clip
        return clip

    # ------------------------------------------------------------------
    def __call__(
        self,
        clip: torch.Tensor,
        silhouette: Optional[torch.Tensor] = None,
        clip_id: str = "",
        make_distorted: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Transform one clip.

        Args:
            clip: ``[T, 3, H, W]``, uint8-valued or already in ``[0, 1]``.
            silhouette: ``[T, 1, H, W]`` binary masks, or None.
            clip_id: the sample id, used to key the deterministic hue shift.
            make_distorted: also return the elastic-distorted twin needed by
                the bias-learning branch.

        Returns:
            ``{"frames": [T,3,h,w]}`` plus ``"frames_distorted"`` and
            ``"silhouette"`` when requested/available.
        """
        clip = self._to_unit_float(clip)
        clip = self._resize_rgb(clip)

        # Fixed per-video appearance nuisance (see the module docstring).
        if self.hue_shift > 0:
            clip = TF.adjust_hue(clip, stable_hue_factor(clip_id, self.hue_shift))

        flip = self.train and random.random() < self.horizontal_flip
        if flip:
            clip = TF.hflip(clip)

        # Branch *after* every appearance decision, so the two clips differ by
        # morphology alone. This is the invariant Eq. 5 depends on.
        distorted = None
        if make_distorted and self.distortion_alpha > 0:
            displacement = elastic_displacement(
                self.image_size, self.distortion_alpha, self.distortion_sigma
            )
            distorted = apply_elastic(clip, displacement)

        out = {"frames": TF.normalize(clip, mean=self.mean, std=self.std)}
        if distorted is not None:
            out["frames_distorted"] = TF.normalize(distorted, mean=self.mean, std=self.std)
        elif make_distorted:
            # alpha == 0 is the documented "no distortion" ablation switch; the
            # branch still runs so the loss stays finite (it becomes ~margin).
            out["frames_distorted"] = out["frames"].clone()

        if self.train and self.random_erasing > 0 and random.random() < self.random_erasing:
            out["frames"] = self._erase(out["frames"])

        if silhouette is not None:
            sil = self._to_unit_float(silhouette)
            sil = TF.resize(sil, list(self.silhouette_size), antialias=True)
            if flip:
                sil = TF.hflip(sil)
            # Keep it genuinely binary: the teacher is "bias-less" precisely
            # because its input carries no intensity information.
            out["silhouette"] = (sil > 0.5).float()

        return out


def build_transform(cfg, train: bool) -> ClipTransform:
    """Build a :class:`ClipTransform` from the ``data`` config block."""
    aug = cfg.get("augmentation", {}) or {}
    return ClipTransform(
        image_size=cfg.get("image_size", DEFAULT_IMAGE_SIZE),
        silhouette_size=cfg.get("silhouette_size", DEFAULT_SILHOUETTE_SIZE),
        train=train,
        horizontal_flip=aug.get("horizontal_flip", 0.5),
        random_crop_pad=aug.get("random_crop_pad", 0),
        random_erasing=aug.get("random_erasing", 0.0),
        hue_shift=aug.get("hue_shift", 0.5),
        distortion_alpha=aug.get("distortion_alpha", 250.0),
        distortion_sigma=aug.get("distortion_sigma", 5.0),
    )
