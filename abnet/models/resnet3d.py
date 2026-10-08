"""ResNet3D video encoder ``S_phi`` (and the shared distortion encoder ``A_phi``).

Vendored from `3D-ResNets-PyTorch <https://github.com/kenshohara/3D-ResNets-PyTorch>`_
(Hara et al., MIT licence), which is the source of the Kinetics-pretrained
ResNet3D-50 the paper names as the backbone: "We use ResNet3D-50 [18] as the
backbone of the video encoder S_phi(.)".

Two deliberate changes from the upstream file:

1. **Temporal strides.** Upstream uses isotropic stride-2 downsampling, so an
   8-frame clip collapses to a single temporal position
   (``8 -> 4`` at the max-pool, then ``4 -> 2 -> 1`` through layers 2-4) and the
   decoder would receive no motion information at all. ABNet feeds 8 frames and
   needs them, so the max-pool and the layer strides become ``(1, 2, 2)``:
   spatial downsampling is untouched, time is preserved. Strides are not
   parameters, so pretrained weights still load unchanged. ``temporal_strides``
   exposes the upstream behaviour if you want it.

2. **Feature extraction instead of classification.** ``forward`` returns the
   ``layer4`` feature map ``F_AB`` of shape ``[B, 2048, T, H', W']`` rather than
   class logits; the ``fc`` layer is dropped. The heads in
   :mod:`abnet.models.decoder` consume that map.

With the paper's 8 x 256 x 128 input this yields ``[B, 2048, 8, 8, 4]``, i.e.
256 spatio-temporal tokens of width 2048.
"""

import os
from functools import partial
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from abnet.utils.logger import get_logger

#: Output channel width of each stage, before the block expansion factor.
BLOCK_INPLANES = [64, 128, 256, 512]

#: ``model_depth -> (block, layers)``.
RESNET_SPEC = {
    10: ("basic", [1, 1, 1, 1]),
    18: ("basic", [2, 2, 2, 2]),
    34: ("basic", [3, 4, 6, 3]),
    50: ("bottleneck", [3, 4, 6, 3]),
    101: ("bottleneck", [3, 4, 23, 3]),
    152: ("bottleneck", [3, 8, 36, 3]),
    200: ("bottleneck", [3, 24, 36, 3]),
}


def conv3x3x3(in_planes: int, out_planes: int, stride=1) -> nn.Conv3d:
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1x1(in_planes: int, out_planes: int, stride=1) -> nn.Conv3d:
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


class ResNet3D(nn.Module):
    """3D ResNet trunk returning a spatio-temporal feature map.

    Args:
        block: ``BasicBlock`` or ``Bottleneck``.
        layers: blocks per stage.
        block_inplanes: stage widths.
        n_input_channels: 3 for RGB.
        conv1_t_size / conv1_t_stride: temporal kernel/stride of the stem.
        no_max_pool: skip the stem max-pool entirely.
        shortcut_type: ``'B'`` (projection, the default) or ``'A'``
            (zero-padded identity), kept for checkpoint compatibility.
        widen_factor: channel multiplier.
        temporal_strides: if False (the default) the max-pool and all stage
            strides become ``(1, 2, 2)``, preserving clip length. True restores
            the upstream isotropic strides.
    """

    def __init__(
        self,
        block,
        layers: Sequence[int],
        block_inplanes: Sequence[int] = BLOCK_INPLANES,
        n_input_channels: int = 3,
        conv1_t_size: int = 7,
        conv1_t_stride: int = 1,
        no_max_pool: bool = False,
        shortcut_type: str = "B",
        widen_factor: float = 1.0,
        temporal_strides: bool = False,
    ):
        super().__init__()
        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool
        self.temporal_strides = temporal_strides

        def stride3(spatial: int):
            t = spatial if temporal_strides else 1
            return (t, spatial, spatial)

        self.conv1 = nn.Conv3d(
            n_input_channels,
            self.in_planes,
            kernel_size=(conv1_t_size, 7, 7),
            stride=(conv1_t_stride, 2, 2),
            padding=(conv1_t_size // 2, 3, 3),
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=stride3(2), padding=1)

        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, block_inplanes[1], layers[1], shortcut_type, stride=stride3(2)
        )
        self.layer3 = self._make_layer(
            block, block_inplanes[2], layers[2], shortcut_type, stride=stride3(2)
        )
        self.layer4 = self._make_layer(
            block, block_inplanes[3], layers[3], shortcut_type, stride=stride3(2)
        )

        self.out_channels = block_inplanes[3] * block.expansion

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(
            out.size(0), planes - out.size(1), out.size(2), out.size(3), out.size(4),
            device=out.device, dtype=out.dtype,
        )
        return torch.cat([out, zero_pads], dim=1)

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        needs_downsample = stride != 1 and stride != (1, 1, 1)
        if needs_downsample or self.in_planes != planes * block.expansion:
            if shortcut_type == "A":
                downsample = partial(
                    self._downsample_basic_block, planes=planes * block.expansion, stride=stride
                )
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion),
                )

        layers = [block(self.in_planes, planes, stride=stride, downsample=downsample)]
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``[B, 3, T, H, W]``.

        Returns:
            ``F_AB`` of shape ``[B, out_channels, T', H', W']``.
        """
        x = self.relu(self.bn1(self.conv1(x)))
        if not self.no_max_pool:
            x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)


def build_resnet3d(model_depth: int = 50, **kwargs) -> ResNet3D:
    """``ResNet3D`` of the requested depth, mirroring upstream's ``generate_model``."""
    if model_depth not in RESNET_SPEC:
        raise ValueError(f"model_depth must be one of {sorted(RESNET_SPEC)}, got {model_depth}")
    kind, layers = RESNET_SPEC[model_depth]
    block = BasicBlock if kind == "basic" else Bottleneck
    return ResNet3D(block, layers, BLOCK_INPLANES, **kwargs)


def _strip_prefixes(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for key, value in state_dict.items():
        for prefix in ("module.", "backbone.", "encoder."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        out[key] = value
    return out


def load_pretrained(
    model: ResNet3D, path: Optional[str], strict: bool = False
) -> List[str]:
    """Load a 3D-ResNets-PyTorch checkpoint into ``model``.

    Upstream checkpoints (``r3d50_K_200ep.pth`` and friends) are dicts with an
    ``arch``/``state_dict``/``epoch`` payload and a ``fc`` classifier sized for
    Kinetics. The classifier is dropped here because ABNet replaces it with its
    own actor and activity heads.

    Returns the list of keys that were not found in the checkpoint. A missing
    file is a warning rather than an error, so the pipeline still runs from
    random init -- but the paper's numbers assume Kinetics initialisation, so
    the warning is loud.
    """
    logger = get_logger()
    if not path:
        logger.warning(
            "model.pretrained is unset: the video encoder starts from random init. "
            "The paper uses a Kinetics-pretrained ResNet3D-50; expect a large drop. "
            "Download r3d50_K_200ep.pth from "
            "https://github.com/kenshohara/3D-ResNets-PyTorch#pre-trained-models"
        )
        return []
    if not os.path.exists(path):
        logger.warning(
            f"model.pretrained not found at '{path}'; starting from random init. "
            f"Download r3d50_K_200ep.pth from "
            f"https://github.com/kenshohara/3D-ResNets-PyTorch#pre-trained-models"
        )
        return []

    payload = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    state_dict = _strip_prefixes(state_dict)
    # The Kinetics classifier has no counterpart in ABNet.
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("fc.")}

    current = model.state_dict()
    compatible, skipped = {}, []
    for key, value in state_dict.items():
        if key in current and current[key].shape == value.shape:
            compatible[key] = value
        else:
            skipped.append(key)

    result = model.load_state_dict(compatible, strict=strict)
    missing = list(result.missing_keys)
    arch = payload.get("arch", "?") if isinstance(payload, dict) else "?"
    logger.info(
        f"loaded pretrained video encoder from {path} (arch={arch}): "
        f"{len(compatible)}/{len(current)} tensors matched"
    )
    if skipped:
        logger.info(f"  skipped {len(skipped)} incompatible tensor(s), e.g. {skipped[:4]}")
    if missing:
        logger.info(f"  {len(missing)} tensor(s) kept at init, e.g. {missing[:4]}")
    return missing
