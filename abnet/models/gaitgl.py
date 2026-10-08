"""GaitGL silhouette encoder ``T_theta`` -- the bias-less teacher.

Vendored from the official GaitGL release
`bb12346/GaitGL <https://github.com/bb12346/GaitGL>`_ (``model/network/vgg_c3d.py``,
class ``C3D_VGG``), cross-checked against the OpenGait reference implementation.
The paper names it directly: "GaitGL [28] for the teacher's silhouette encoder
T_theta(.)".

Architecture, following the GaitGL paper (ICCV 2021):

* a plain 3D conv stem;
* **LTA** -- local temporal aggregation, a ``(3,1,1)`` conv with temporal
  stride 3, which compresses time without the information loss of pooling;
* four **GLConv** blocks, each running a global 3D conv over the whole frame in
  parallel with a local 3D conv applied to horizontal body partitions, so
  whole-body and part-level gait cues are captured together. The final block
  uses ``fm_sign=True``, concatenating the two streams along height instead of
  summing them;
* temporal max pooling, then **GeM** horizontal-pyramid pooling into 64 bins;
* a part-wise separate FC, giving a ``[B, 256, 64]`` part-feature tensor.

What this file adds beyond upstream: an **identity head**. Upstream GaitGL
returns embeddings only (``return feature, None``), but Eq. 1 distils a
*probability distribution* ``y_T``, so a classifier is required. The head
follows GaitGL's own ``Bn_head`` variant -- BatchNorm1d over the channel axis,
then a part-wise separate FC to the identity space -- and the per-part logits
are averaged into one distribution per clip.

The teacher is trained by ``train_teacher.py`` (stage 0) and frozen thereafter;
it is never used at inference, matching "we only use silhouette during training
and it is not required for inference".
"""

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

#: Upstream ``_set_channels``.
GAITGL_CHANNELS = [32, 64, 128, 256]

#: Upstream ``bin_numgl = [32 * 2]``; the final GLConv doubles the height via
#: ``fm_sign=True``, so the horizontal pyramid has 64 bins.
GAITGL_NUM_BINS = 64


def gem(x: torch.Tensor, p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Generalised-mean pooling over the last axis."""
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (1, x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    """GeM pooling with a learnable exponent (upstream default ``p = 6.5``)."""

    def __init__(self, p: float = 6.5, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p.data.item():.4f}, eps={self.eps})"


class BasicConv3d(nn.Module):
    """3x3x3 conv + leaky ReLU."""

    def __init__(self, inplanes: int, planes: int, dilation: int = 1, bias: bool = False):
        super().__init__()
        self.conv1 = nn.Conv3d(
            inplanes, planes, kernel_size=(3, 3, 3), bias=bias,
            dilation=(dilation, 1, 1), padding=(dilation, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(self.conv1(x), inplace=True)


class GLConv(nn.Module):
    """Global-local convolution (upstream ``BasicConv3d_p``).

    The global branch convolves the whole feature map; the local branch splits
    it into ``p`` horizontal strips and convolves each independently with shared
    weights, which is what gives GaitGL its part-level sensitivity.

    Args:
        p: number of horizontal partitions.
        fm_sign: when False the branches are summed; when True they are
            concatenated along the height axis, doubling it.
    """

    def __init__(
        self, inplanes: int, planes: int, kernel: int = 3, bias: bool = False,
        p: int = 2, fm_sign: bool = False,
    ):
        super().__init__()
        self.p = p
        self.fm_sign = fm_sign
        pad = (kernel - 1) // 2
        self.convdl = nn.Conv3d(
            inplanes, planes, kernel_size=(kernel, kernel, kernel), bias=bias,
            padding=(pad, pad, pad),
        )
        self.convdg = nn.Conv3d(
            inplanes, planes, kernel_size=(kernel, kernel, kernel), bias=bias,
            padding=(pad, pad, pad),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height = x.size(3)
        scale = height // self.p
        local = torch.cat(
            [self.convdl(x[:, :, :, i * scale : (i + 1) * scale, :]) for i in range(self.p)],
            dim=3,
        )
        local = F.leaky_relu(local, inplace=True)
        glob = F.leaky_relu(self.convdg(x), inplace=True)
        if self.fm_sign:
            return torch.cat((glob, local), dim=3)
        return glob + local


class LocalTemporalAggregation(nn.Module):
    """LTA: ``(3,1,1)`` conv with temporal stride 3."""

    def __init__(self, inplanes: int, planes: int, bias: bool = False):
        super().__init__()
        self.conv1 = nn.Conv3d(
            inplanes, planes, kernel_size=(3, 1, 1), stride=(3, 1, 1), bias=bias, padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(self.conv1(x), inplace=True)


class SeparateFCs(nn.Module):
    """One independent linear map per horizontal bin.

    Upstream implements this as a bare ``ParameterList`` of a
    ``[bins, in, out]`` tensor plus a ``matmul``; wrapping it keeps the
    state-dict readable and lets the identity head reuse it.
    """

    def __init__(self, num_bins: int, in_channels: int, out_channels: int):
        super().__init__()
        self.weight = nn.Parameter(
            nn.init.xavier_uniform_(torch.zeros(num_bins, in_channels, out_channels))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, C, P] -> [B, C_out, P]``."""
        # [B, C, P] -> [P, B, C] -> [P, B, C_out] -> [B, C_out, P]
        out = x.permute(2, 0, 1).contiguous().matmul(self.weight)
        return out.permute(1, 2, 0).contiguous()


class GaitGL(nn.Module):
    """Silhouette encoder plus an identity classifier.

    Args:
        num_classes: size of the identity label space. Must match the student's,
            since Eq. 1 compares the two distributions element-wise.
        channels: stage widths.
        num_bins: horizontal-pyramid bins.
        halving: number of horizontal partitions in the local branch
            (upstream CASIA-B config uses ``p = 2``).

    Input is ``[B, 1, T, H, W]`` binary silhouettes; ``tools/extract_silhouettes.py``
    writes them at 64x44, GaitGL's native size.
    """

    def __init__(
        self,
        num_classes: int,
        channels: Sequence[int] = GAITGL_CHANNELS,
        num_bins: int = GAITGL_NUM_BINS,
        halving: int = 2,
    ):
        super().__init__()
        c = list(channels)
        self.num_classes = num_classes
        self.num_bins = num_bins
        self.feature_dim = c[3]

        self.conv1 = BasicConv3d(1, c[0])
        self.lta = LocalTemporalAggregation(c[0], c[0])

        self.glconv1 = GLConv(c[0], c[1], p=halving, fm_sign=False)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.glconv2a = GLConv(c[1], c[2], p=halving, fm_sign=False)
        self.glconv2b = GLConv(c[2], c[2], p=halving, fm_sign=False)

        self.glconv3a = GLConv(c[2], c[3], p=halving, fm_sign=False)
        self.glconv3b = GLConv(c[3], c[3], p=halving, fm_sign=True)

        self.gem = GeM()
        self.head = SeparateFCs(num_bins, c[3], c[3])

        # Identity head (see the module docstring): not part of upstream GaitGL,
        # required to produce y_T for Eq. 1.
        self.bn = nn.BatchNorm1d(c[3])
        self.classifier = SeparateFCs(num_bins, c[3], num_classes)

        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)

    @staticmethod
    def _pad_short_clips(sils: torch.Tensor) -> torch.Tensor:
        """LTA needs at least 3 frames; repeat short clips as upstream does."""
        frames = sils.size(2)
        if frames >= 3:
            return sils
        repeat = 3 if frames == 1 else 2
        return sils.repeat(1, 1, repeat, 1, 1)

    def forward(self, silhouettes: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            silhouettes: ``[B, 1, T, H, W]`` or ``[B, T, 1, H, W]`` binary masks.

        Returns:
            ``{"logits": [B, num_classes], "embedding": [B, C*P],
            "part_logits": [B, num_classes, P], "part_embedding": [B, C, P]}``.
        """
        if silhouettes.dim() != 5:
            raise ValueError(
                f"expected a 5D silhouette tensor, got {tuple(silhouettes.shape)}"
            )
        # Accept the dataloader's [B, T, 1, H, W] as well as GaitGL's [B, 1, T, H, W].
        if silhouettes.size(1) != 1 and silhouettes.size(2) == 1:
            silhouettes = silhouettes.transpose(1, 2)

        x = self._pad_short_clips(silhouettes)

        x = self.conv1(x)
        x = self.lta(x)

        x = self.glconv1(x)
        x = self.maxpool(x)

        x = self.glconv2a(x)
        x = self.glconv2b(x)

        x = self.glconv3a(x)
        x = self.glconv3b(x)  # [B, C, T, H, W]

        # Temporal pooling: "set pooling" by max over time, as in GaitGL.
        x = torch.max(x, dim=2)[0]  # [B, C, H, W]

        batch, channels = x.shape[:2]
        binned = x.view(batch, channels, self.num_bins, -1).contiguous()
        part_feature = self.gem(binned).squeeze(-1)  # [B, C, P]

        part_embedding = self.head(part_feature)  # [B, C, P]
        normed = self.bn(part_embedding)  # BatchNorm1d over the channel axis
        part_logits = self.classifier(normed)  # [B, num_classes, P]

        return {
            # One distribution per clip: average the per-part logits. Averaging
            # logits (not probabilities) keeps this a plain linear readout of
            # the part features, so the softmax in Eq. 1 is applied once.
            "logits": part_logits.mean(dim=-1),
            "part_logits": part_logits,
            "part_embedding": normed,
            "embedding": normed.flatten(1),
            "triplet_embedding": part_embedding,
        }


def build_gaitgl(num_classes: int, cfg: Optional[Dict] = None) -> GaitGL:
    cfg = cfg or {}
    return GaitGL(
        num_classes=num_classes,
        channels=cfg.get("channels", GAITGL_CHANNELS),
        num_bins=cfg.get("num_bins", GAITGL_NUM_BINS),
        halving=cfg.get("halving", 2),
    )
