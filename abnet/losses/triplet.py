"""Batch-hard triplet loss (Eq. 4)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def euclidean_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Pairwise Euclidean distance between rows of ``x`` ``[m, d]`` and ``y`` ``[n, d]``."""
    m, n = x.size(0), y.size(0)
    xx = x.pow(2).sum(1, keepdim=True).expand(m, n)
    yy = y.pow(2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy - 2 * x @ y.t()
    return dist.clamp(min=1e-12).sqrt()


class TripletLoss(nn.Module):
    """L_tri = max(D(f_a, f_p) - D(f_a, f_n) + m, 0).

    Positives and negatives are mined within the batch (the hardest of each) --
    the paper's "f_p and f_n are the positive and negative features for an
    anchor feature f_a within the same batch" -- which is why training uses a
    P-identities x K-clips sampler. ``m`` is 0.3.
    """

    def __init__(self, margin: float = 0.3, normalize_feature: bool = False,
                 soft_margin: bool = False):
        super().__init__()
        self.margin = margin
        self.normalize_feature = normalize_feature
        self.soft_margin = soft_margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: ``[B, D]`` biometrics features ``f_bb``.
            labels: ``[B]`` identity labels.
        """
        if self.normalize_feature:
            features = F.normalize(features, p=2, dim=-1)

        dist = euclidean_dist(features, features)
        same = labels[:, None].eq(labels[None, :])
        diff = ~same
        eye = torch.eye(dist.size(0), dtype=torch.bool, device=dist.device)
        positive_mask = same & ~eye

        has_pos = positive_mask.any(dim=1)
        has_neg = diff.any(dim=1)
        valid = has_pos & has_neg
        if not valid.any():
            # Every identity is unique in this batch: nothing to rank.
            return features.sum() * 0.0

        neg_inf = torch.finfo(dist.dtype).min
        pos_inf = torch.finfo(dist.dtype).max
        dist_ap = dist.masked_fill(~positive_mask, neg_inf).max(dim=1).values
        dist_an = dist.masked_fill(~diff, pos_inf).min(dim=1).values

        dist_ap, dist_an = dist_ap[valid], dist_an[valid]
        if self.soft_margin:
            return F.softplus(dist_ap - dist_an).mean()
        target = torch.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, target)
