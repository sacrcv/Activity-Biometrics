"""Identity classification head."""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BNNeck(nn.Module):
    """BNNeck classification head, the standard video re-ID setup.

    The pre-BN feature feeds the triplet term of Eq. 4, the post-BN feature
    feeds the cross-entropy term (and is what retrieval uses at test time).
    Separating the two is what lets a metric loss and a classification loss
    share one embedding without fighting over its scale.
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        dropout: float = 0.0,
        cosine_classifier: bool = False,
        cosine_scale: float = 16.0,
    ):
        super().__init__()
        self.bottleneck = nn.BatchNorm1d(in_dim)
        nn.init.constant_(self.bottleneck.weight, 1.0)
        nn.init.zeros_(self.bottleneck.bias)
        # Classic BNNeck: the shift is frozen so the feature stays centred.
        self.bottleneck.bias.requires_grad_(False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.cosine_classifier = cosine_classifier
        self.cosine_scale = cosine_scale

        self.classifier = nn.Linear(in_dim, num_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: ``[B, in_dim]`` pooled feature.

        Returns:
            ``(logits [B, num_classes], bn_feature [B, in_dim])``.
        """
        feat = self.bottleneck(x)
        logits_in = self.dropout(feat)
        if self.cosine_classifier:
            logits = self.cosine_scale * F.linear(
                F.normalize(logits_in, dim=-1), F.normalize(self.classifier.weight, dim=-1)
            )
        else:
            logits = self.classifier(logits_in)
        return logits, feat
