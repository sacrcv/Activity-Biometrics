"""The actor head ``C^B`` and activity head ``C^A``.

Both are transformer decoders over the spatio-temporal tokens of ``F_AB``.
Section 3.1 describes the actor head:

    "One split segment of the extracted feature F_AB is fed to the actor head
    C^B, which contains D^B_omega that is a standard transformer decoder. We get
    actor feature F_BT from D^B_omega, which contains biometrics feature f_bb
    and appearance feature f_ba. D^B_omega uses self-attention to process the
    input sequence and then projects the attention output into f_bb and f_ba
    using separate linear layers."

The paper does not fix the decoder's width, depth or query count, so this is a
documented default rather than a transcription. The design is the standard
DETR-style decoder: a small set of learnable query tokens self-attend among
themselves, cross-attend to the flattened ``F_AB`` tokens, and are then
mean-pooled into one vector per clip.
That satisfies the paper's description, keeps the head small (~8M parameters
across both heads), and -- importantly -- makes the output dimension independent
of the input resolution, so a 256x128 clip and a 224x224 clip yield the same
feature size.

Positional information is sinusoidal over the flattened token index. A learnable
table would need interpolation whenever the clip length or crop size changes;
sinusoidal encodings simply work at any token count, which matters because
Charades-AB and NTU RGB-AB are configured with different clip handling.
"""

import math
from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn


def sinusoidal_encoding(num_tokens: int, dim: int, device, dtype) -> torch.Tensor:
    """Standard sinusoidal positional encoding, ``[1, num_tokens, dim]``."""
    position = torch.arange(num_tokens, device=device, dtype=torch.float32).unsqueeze(1)
    index = torch.arange(0, dim, 2, device=device, dtype=torch.float32)
    divisor = torch.exp(-math.log(10000.0) * index / dim)
    table = torch.zeros(num_tokens, dim, device=device, dtype=torch.float32)
    table[:, 0::2] = torch.sin(position * divisor)
    table[:, 1::2] = torch.cos(position * divisor)[:, : table[:, 1::2].shape[1]]
    return table.unsqueeze(0).to(dtype)


class QueryDecoder(nn.Module):
    """Learnable queries cross-attending to a token sequence.

    Args:
        in_dim: width of the incoming tokens (one half of ``F_AB``'s channels
            when ``split_mode="channel"``).
        d_model: decoder width.
        num_queries: number of learnable query tokens.
        num_layers: decoder depth.
        nhead: attention heads.
        dim_feedforward: FFN width.
        dropout: dropout inside the decoder.
    """

    def __init__(
        self,
        in_dim: int,
        d_model: int = 512,
        num_queries: int = 16,
        num_layers: int = 2,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries

        self.input_proj = nn.Linear(in_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.query = nn.Parameter(torch.zeros(1, num_queries, d_model))
        nn.init.trunc_normal_(self.query, std=0.02)

        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-LN: more stable without a warmup-heavy schedule
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: ``[B, N, in_dim]`` spatio-temporal tokens.

        Returns:
            ``[B, d_model]`` pooled decoder output.
        """
        memory = self.input_norm(self.input_proj(tokens))
        memory = memory + sinusoidal_encoding(
            memory.size(1), self.d_model, memory.device, memory.dtype
        )
        query = self.query.expand(memory.size(0), -1, -1)
        out = self.decoder(tgt=query, memory=memory)
        return self.out_norm(out).mean(dim=1)


class ActorHead(nn.Module):
    """``C^B``: decodes ``F_BT`` and splits it into ``f_bb`` and ``f_ba``.

    The two "separate linear layers" of Section 3.1 are ``to_biometrics`` and
    ``to_appearance``. They are deliberately *not* tied and not orthogonalised:
    the disentanglement pressure comes entirely from the losses (distillation
    pulls ``f_bb`` towards the silhouette teacher, Eq. 5 pushes ``f_ba`` and
    ``f_bb`` in opposite directions under distortion), not from an architectural
    constraint.
    """

    def __init__(
        self,
        in_dim: int,
        feature_dim: int = 256,
        d_model: int = 512,
        num_queries: int = 16,
        num_layers: int = 2,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.decoder = QueryDecoder(
            in_dim=in_dim,
            d_model=d_model,
            num_queries=num_queries,
            num_layers=num_layers,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.to_biometrics = nn.Linear(d_model, feature_dim)
        self.to_appearance = nn.Linear(d_model, feature_dim)
        self.feature_dim = feature_dim
        self.d_model = d_model

    def forward(self, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        actor_feature = self.decoder(tokens)  # F_BT
        return {
            "actor_feature": actor_feature,
            "biometrics": self.to_biometrics(actor_feature),
            "appearance": self.to_appearance(actor_feature),
        }


class ActivityHead(nn.Module):
    """``C^A``: decodes ``F_Ac`` and classifies the activity.

    ``F_Ac`` does double duty -- it is the input to ``L_Ac`` during training and
    the activity prior concatenated onto ``f_bb`` at inference (Section 3.2).
    """

    def __init__(
        self,
        in_dim: int,
        num_actions: int,
        d_model: int = 512,
        num_queries: int = 16,
        num_layers: int = 2,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.decoder = QueryDecoder(
            in_dim=in_dim,
            d_model=d_model,
            num_queries=num_queries,
            num_layers=num_layers,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.classifier = nn.Linear(d_model, num_actions)
        self.d_model = d_model

    def forward(self, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        activity_feature = self.decoder(tokens)  # F_Ac
        return {
            "activity_feature": activity_feature,
            "action_logits": self.classifier(activity_feature),
        }


def flatten_tokens(feature_map: torch.Tensor) -> torch.Tensor:
    """``[B, C, T, H, W] -> [B, T*H*W, C]``."""
    batch, channels = feature_map.shape[:2]
    return feature_map.reshape(batch, channels, -1).transpose(1, 2).contiguous()


def split_feature(
    feature_map: torch.Tensor, mode: str = "channel"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split ``F_AB`` into the two "segments" of Section 3.

    The paper says only that "the spatio-temporal feature F_AB is split into two
    segments and are passed to the actor head C^B ... as well as the activity
    head C^A", without saying along which axis.

    * ``channel`` (default): halve the channel axis. Both heads keep the full
      spatio-temporal extent, which the activity head needs -- it has to see the
      whole motion -- and the parameter cost is balanced.
    * ``temporal``: halve the clip. Cheaper, but gives each head 4 frames
      instead of 8, which measurably hurts the activity branch.
    * ``shared``: no split; both heads see all of ``F_AB``. Useful as a control.

    Returns ``(actor_tokens, activity_tokens)``.
    """
    if mode == "shared":
        tokens = flatten_tokens(feature_map)
        return tokens, tokens
    if mode == "channel":
        channels = feature_map.size(1)
        if channels % 2 != 0:
            raise ValueError(f"channel split needs an even channel count, got {channels}")
        half = channels // 2
        actor = flatten_tokens(feature_map[:, :half])
        activity = flatten_tokens(feature_map[:, half:])
        return actor, activity
    if mode == "temporal":
        frames = feature_map.size(2)
        if frames < 2:
            raise ValueError(
                f"temporal split needs at least 2 temporal positions, got {frames}; "
                f"the backbone must preserve clip length (model.temporal_strides=false)"
            )
        half = frames // 2
        actor = flatten_tokens(feature_map[:, :, :half])
        activity = flatten_tokens(feature_map[:, :, half:])
        return actor, activity
    raise ValueError(f"split_mode must be 'channel', 'temporal' or 'shared', got '{mode}'")


def split_dim(channels: int, mode: str = "channel") -> int:
    """Token width each head receives under a given ``split_mode``."""
    if mode == "channel":
        return channels // 2
    if mode in ("temporal", "shared"):
        return channels
    raise ValueError(f"split_mode must be 'channel', 'temporal' or 'shared', got '{mode}'")


__all__: Sequence[str] = [
    "QueryDecoder",
    "ActorHead",
    "ActivityHead",
    "flatten_tokens",
    "split_feature",
    "split_dim",
    "sinusoidal_encoding",
]
