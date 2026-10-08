"""ABNet: Activity Biometrics Network (Figure 2).

Forward pass, following Section 3:

1. ``S_phi`` (ResNet3D-50) encodes the RGB clip into ``F_AB``.
2. ``F_AB`` is split into two segments, one per head.
3. ``C^B`` (actor head) produces ``F_BT`` and projects it into the biometrics
   feature ``f_bb`` and the appearance feature ``f_ba``.
4. ``C^A`` (activity head) produces ``F_Ac`` and the activity logits.
5. The distortion branch ``A`` re-runs steps 1-3 on an elastically distorted
   copy of the clip to obtain ``f_bb^D`` and ``f_ba^D``. "this distortion
   network A, which is identical to M and shares weights" -- so it is literally
   a second call through the same modules, with no parameters of its own. Its
   activity head is not used: "Since this branch is designed for bias-learning,
   thus the activity head C^DA of A is not utilized."

At inference only the ``M`` branch runs, and the retrieval feature is
``concat(F_Ac, f_bb)``: "During inference the activity feature F_Ac is
concatenated with the biometrics feature f_bb that acts as the activity prior."

The teacher lives outside this module. It consumes silhouettes, not RGB, and is
frozen, so :mod:`abnet.engine.trainer` owns it and feeds its logits into
``L_KD``. Keeping it out of ``ABNetModel`` is what makes the inference path
provably silhouette-free.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from abnet.models.decoder import ActivityHead, ActorHead, split_dim, split_feature
from abnet.models.heads import BNNeck
from abnet.models.resnet3d import build_resnet3d, load_pretrained
from abnet.utils.logger import get_logger

#: Features that :func:`ABNetModel.extract_features` can return for retrieval.
FEATURE_CHOICES = ("fused", "biometrics", "appearance", "activity", "actor")


class ABNetModel(nn.Module):
    """The full network ``M``.

    Args:
        num_identities: size of the identity label space (train split).
        num_actions: number of activity classes.
        backbone_depth: ResNet3D depth; the paper uses 50.
        pretrained: path to a 3D-ResNets-PyTorch checkpoint.
        temporal_strides: see :class:`abnet.models.resnet3d.ResNet3D`. Leave
            False so the 8-frame clip survives to the heads.
        split_mode: how ``F_AB`` is divided between the heads.
        feature_dim: width of ``f_bb`` / ``f_ba``.
        decoder: kwargs forwarded to both heads.
        bnneck_dropout: dropout before the identity classifier.
        cosine_classifier: use a cosine-margin identity classifier instead of a
            plain linear one.
    """

    def __init__(
        self,
        num_identities: int,
        num_actions: int,
        backbone_depth: int = 50,
        pretrained: Optional[str] = None,
        temporal_strides: bool = False,
        split_mode: str = "channel",
        feature_dim: int = 256,
        decoder: Optional[Dict] = None,
        bnneck_dropout: float = 0.0,
        cosine_classifier: bool = False,
        conv1_t_stride: int = 1,
    ):
        super().__init__()
        decoder = dict(decoder or {})
        self.split_mode = split_mode
        self.feature_dim = feature_dim
        self.num_identities = num_identities
        self.num_actions = num_actions

        self.video_encoder = build_resnet3d(
            backbone_depth,
            temporal_strides=temporal_strides,
            conv1_t_stride=conv1_t_stride,
        )
        load_pretrained(self.video_encoder, pretrained)

        token_dim = split_dim(self.video_encoder.out_channels, split_mode)

        self.actor_head = ActorHead(in_dim=token_dim, feature_dim=feature_dim, **decoder)
        self.activity_head = ActivityHead(
            in_dim=token_dim, num_actions=num_actions, **decoder
        )

        # Identity classification runs off f_bb through a BNNeck: the pre-BN
        # feature feeds the triplet term of Eq. 4, the post-BN logits feed the
        # cross-entropy term and the distillation target of Eq. 1.
        self.identity_head = BNNeck(
            in_dim=feature_dim,
            num_classes=num_identities,
            dropout=bnneck_dropout,
            cosine_classifier=cosine_classifier,
        )

        self.activity_prior_dim = self.activity_head.d_model
        get_logger().info(
            f"ABNet: F_AB={self.video_encoder.out_channels}ch, split='{split_mode}' "
            f"-> {token_dim}ch per head, f_bb/f_ba={feature_dim}d, "
            f"F_Ac={self.activity_prior_dim}d, "
            f"retrieval feature={self.retrieval_dim}d, "
            f"{num_identities} identities, {num_actions} actions"
        )

    @property
    def retrieval_dim(self) -> int:
        """Width of the default (``fused``) retrieval feature."""
        return self.activity_prior_dim + self.feature_dim

    # ------------------------------------------------------------------
    def encode(self, frames: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run one branch: clip -> ``F_AB`` -> both heads.

        This is the single code path used by ``M``, by the shared-weight
        distortion branch ``A``, and by inference, which is what makes "A is
        identical to M and shares weights" true by construction rather than by
        convention.

        Args:
            frames: ``[B, T, 3, H, W]`` (dataloader order) or
                ``[B, 3, T, H, W]`` (conv order).
        """
        if frames.dim() != 5:
            raise ValueError(f"expected a 5D clip tensor, got {tuple(frames.shape)}")
        # The dataloader yields [B, T, C, H, W]; Conv3d wants [B, C, T, H, W].
        if frames.size(2) == 3 and frames.size(1) != 3:
            frames = frames.transpose(1, 2)

        feature_map = self.video_encoder(frames)  # F_AB
        actor_tokens, activity_tokens = split_feature(feature_map, self.split_mode)

        actor = self.actor_head(actor_tokens)
        activity = self.activity_head(activity_tokens)

        identity_logits, biometrics_bn = self.identity_head(actor["biometrics"])
        return {
            "feature_map": feature_map,
            "actor_feature": actor["actor_feature"],      # F_BT
            "biometrics": actor["biometrics"],            # f_bb (pre-BN)
            "biometrics_bn": biometrics_bn,               # f_bb (post-BN)
            "appearance": actor["appearance"],            # f_ba
            "identity_logits": identity_logits,           # y_S
            "activity_feature": activity["activity_feature"],  # F_Ac
            "action_logits": activity["action_logits"],
        }

    def forward(
        self, frames: torch.Tensor, frames_distorted: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Full training forward.

        Args:
            frames: the clip ``v``.
            frames_distorted: the elastically distorted clip ``v_hat``. When
                given, the shared-weight distortion branch runs and
                ``biometrics_distorted`` / ``appearance_distorted`` are
                returned for Eq. 5.
        """
        out = self.encode(frames)

        if frames_distorted is not None:
            # Same weights, second input. The activity head's output is
            # computed but discarded -- C^DA is unused by design.
            distorted = self.encode(frames_distorted)
            out["biometrics_distorted"] = distorted["biometrics"]
            out["appearance_distorted"] = distorted["appearance"]
        return out

    # ------------------------------------------------------------------
    @torch.no_grad()
    def extract_features(
        self, frames: torch.Tensor, feature: str = "fused", normalize: bool = True
    ) -> torch.Tensor:
        """Retrieval feature for a batch of clips.

        ``fused`` is the paper's inference feature, ``concat(F_Ac, f_bb)``. The
        others exist to reproduce the "performance of disentangled features"
        discussion: ``appearance`` should retrieve *badly*, which is what
        demonstrates that the disentanglement worked.
        """
        if feature not in FEATURE_CHOICES:
            raise ValueError(f"feature must be one of {FEATURE_CHOICES}, got '{feature}'")

        out = self.encode(frames)
        if feature == "fused":
            # The activity prior is concatenated, so each part is normalised
            # first; otherwise whichever block has the larger norm would
            # dominate the Euclidean distance.
            parts = [
                F.normalize(out["activity_feature"], dim=-1),
                F.normalize(out["biometrics_bn"], dim=-1),
            ]
            vector = torch.cat(parts, dim=-1)
        elif feature == "biometrics":
            vector = out["biometrics_bn"]
        elif feature == "appearance":
            vector = out["appearance"]
        elif feature == "activity":
            vector = out["activity_feature"]
        else:  # actor
            vector = out["actor_feature"]

        return F.normalize(vector, dim=-1) if normalize else vector

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg, num_identities: int, num_actions: int) -> "ABNetModel":
        """Build from the ``model`` config block.

        The label-space sizes come from the data, not the config, so a config
        cannot silently disagree with the annotation file it is paired with.
        """
        model_cfg = cfg.model
        decoder_cfg = model_cfg.get("decoder", {}) or {}
        return cls(
            num_identities=num_identities,
            num_actions=num_actions,
            backbone_depth=model_cfg.get("backbone_depth", 50),
            pretrained=model_cfg.get("pretrained", None),
            temporal_strides=model_cfg.get("temporal_strides", False),
            split_mode=model_cfg.get("split_mode", "channel"),
            feature_dim=model_cfg.get("feature_dim", 256),
            bnneck_dropout=model_cfg.get("bnneck_dropout", 0.0),
            cosine_classifier=model_cfg.get("cosine_classifier", False),
            conv1_t_stride=model_cfg.get("conv1_t_stride", 1),
            decoder=dict(
                d_model=decoder_cfg.get("d_model", 512),
                num_queries=decoder_cfg.get("num_queries", 16),
                num_layers=decoder_cfg.get("num_layers", 2),
                nhead=decoder_cfg.get("nhead", 8),
                dim_feedforward=decoder_cfg.get("dim_feedforward", 2048),
                dropout=decoder_cfg.get("dropout", 0.1),
            ),
        )
