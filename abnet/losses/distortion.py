"""Bias learning through biometrics distortion (Eq. 5).

    L_Dis = max(D(f_ba, f_ba^D) - D(f_bb, f_bb^D) + m, 0)

The distortion network ``A`` is ``M`` itself, run a second time on an elastically
distorted copy of the clip. Elastic transform scrambles the morphology of the
person -- their shape, proportions, gait -- while leaving colour, texture and
background intact. So relative to the original clip:

* the appearance features ``f_ba`` and ``f_ba^D`` describe the *same* clothing
  and scene, and are a **positive** pair that should be pulled together;
* the biometrics features ``f_bb`` and ``f_bb^D`` describe *different* bodies,
  and are a **hard negative** pair that should be pushed apart.

Minimising Eq. 5 therefore drives ``f_ba`` to encode only what distortion
preserves (appearance) and ``f_bb`` to encode only what distortion destroys
(biometrics) -- which is the disentanglement, obtained without any extra
parameters or labels.

Sign convention is worth stating because it is easy to invert: the positive
distance is the term being *minimised* and the negative distance the term being
*maximised*, so the positive distance comes first inside the hinge, exactly as
in the triplet loss of Eq. 4.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistortionLoss(nn.Module):
    """Margin loss over the distorted/original feature pairs.

    Args:
        margin: ``m`` in Eq. 5. The paper sets the triplet margin to 0.3 and
            reuses the same symbol for this contrastive margin.
        normalize_feature: L2-normalise before measuring distance. Off by
            default so ``D`` is the plain Euclidean distance the paper names;
            turning it on bounds the loss, which can help if the raw feature
            norms drift early in training.
    """

    def __init__(self, margin: float = 0.3, normalize_feature: bool = False):
        super().__init__()
        self.margin = float(margin)
        self.normalize_feature = normalize_feature

    def forward(
        self,
        biometrics: torch.Tensor,
        biometrics_distorted: torch.Tensor,
        appearance: torch.Tensor,
        appearance_distorted: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            biometrics: ``f_bb`` ``[B, D]``.
            biometrics_distorted: ``f_bb^D`` ``[B, D]``.
            appearance: ``f_ba`` ``[B, D]``.
            appearance_distorted: ``f_ba^D`` ``[B, D]``.

        Returns:
            Scalar loss.
        """
        if self.normalize_feature:
            biometrics = F.normalize(biometrics, p=2, dim=-1)
            biometrics_distorted = F.normalize(biometrics_distorted, p=2, dim=-1)
            appearance = F.normalize(appearance, p=2, dim=-1)
            appearance_distorted = F.normalize(appearance_distorted, p=2, dim=-1)

        # Row-wise distances: pairing is by construction (clip i with its own
        # distorted copy), so no in-batch mining is involved here.
        dist_positive = torch.norm(appearance - appearance_distorted, p=2, dim=-1)
        dist_negative = torch.norm(biometrics - biometrics_distorted, p=2, dim=-1)

        return F.relu(dist_positive - dist_negative + self.margin).mean()
