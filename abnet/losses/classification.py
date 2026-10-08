"""Identity and activity classification losses (Eq. 2, 3)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLabelSmooth(nn.Module):
    """L_ce = -y log(y_hat) (Eq. 3), optionally with label smoothing.

    ``epsilon = 0`` is plain cross entropy, which is what the paper specifies.
    """

    def __init__(self, num_classes: int, epsilon: float = 0.0):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.epsilon <= 0:
            return F.cross_entropy(logits, targets)
        log_probs = F.log_softmax(logits, dim=1)
        with torch.no_grad():
            smooth = torch.full_like(log_probs, self.epsilon / self.num_classes)
            smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.epsilon + self.epsilon / self.num_classes)
        return (-smooth * log_probs).sum(dim=1).mean()


class ActionLoss(nn.Module):
    """L_Ac over the activity head's feature F_Ac (Section 3.2).

    "C^A is trained using L_Ac which is a standard cross-entropy loss for the
    activity labels regardless of the actor labels."

    ``mode="ce"`` for single-label datasets (NTU RGB-AB), ``mode="bce"`` for
    multi-label ones (Charades-AB averages 6.8 activities per video, so its
    targets are a multi-hot vector).
    """

    def __init__(self, num_classes: int, mode: str = "ce", epsilon: float = 0.0):
        super().__init__()
        if mode not in ("ce", "bce"):
            raise ValueError(f"action loss mode must be 'ce' or 'bce', got '{mode}'")
        self.mode = mode
        self.num_classes = num_classes
        self.ce = CrossEntropyLabelSmooth(num_classes, epsilon)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: ``[B, num_classes]``.
            targets: ``[B]`` int labels for ``ce``, ``[B, num_classes]``
                multi-hot float for ``bce``. Entries ``< 0`` (``ce``) or
                all-zero rows (``bce``) are treated as unlabelled and skipped.
        """
        if self.mode == "ce":
            valid = targets >= 0
            if not valid.any():
                return logits.sum() * 0.0
            return self.ce(logits[valid], targets[valid])

        targets = targets.to(logits.dtype)
        valid = targets.sum(dim=1) > 0
        if not valid.any():
            return logits.sum() * 0.0
        return F.binary_cross_entropy_with_logits(logits[valid], targets[valid])
