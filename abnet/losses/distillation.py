"""Bias-less distillation from the silhouette teacher (Eq. 1).

    L_KD = tau^2 * KL(y_T || y_S)

where ``y_T`` and ``y_S`` are the identity distributions of the teacher
(GaitGL on binary silhouettes) and of ABNet's biometrics branch, and ``tau``
controls the softness of the teacher's output.

The teacher is called "bias-less" because its only input is a binary
silhouette, so it cannot have learned clothing colour, background or any other
appearance cue. Pulling the student's biometrics logits towards it is therefore
what pushes appearance information *out* of ``f_bb`` and into ``f_ba``.

Two details that matter in practice:

* The ``tau^2`` factor is written explicitly in Eq. 1. It compensates for the
  ``1/tau^2`` shrinkage that softening induces in the KL gradient, so the term
  stays commensurate with the un-softened cross-entropy of Eq. 3 as ``tau``
  changes.
* ``KL(y_T || y_S)`` has the teacher first. PyTorch's ``kl_div`` expects
  ``input`` to be the log-distribution of the *second* argument, so the call
  below is ``kl_div(log y_S, y_T)``; getting that backwards trains the student
  on the wrong asymmetry and quietly costs accuracy.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """Temperature-scaled KL divergence from a teacher's logits.

    Args:
        temperature: ``tau`` in Eq. 1.
        scale_by_temperature_squared: apply the ``tau^2`` factor. On by
            default because Eq. 1 states it; exposed so the ablation is
            reachable without editing code.
    """

    def __init__(self, temperature: float = 4.0, scale_by_temperature_squared: bool = True):
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"distillation temperature must be > 0, got {temperature}")
        self.temperature = float(temperature)
        self.scale_by_temperature_squared = scale_by_temperature_squared

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            student_logits: ``[B, C]`` identity logits from ABNet.
            teacher_logits: ``[B, C]`` identity logits from the frozen teacher.
            mask: optional ``[B]`` float/bool mask; rows that are 0 are
                excluded. Used to skip clips with no silhouette available.

        Returns:
            Scalar loss. Zero (with a live gradient path) when no row is valid.
        """
        if student_logits.shape != teacher_logits.shape:
            raise ValueError(
                f"student logits {tuple(student_logits.shape)} and teacher logits "
                f"{tuple(teacher_logits.shape)} must have the same shape; the teacher "
                f"must be trained on the same identity label space as the student"
            )

        if mask is not None:
            keep = mask.bool().reshape(-1)
            if not keep.any():
                return student_logits.sum() * 0.0
            student_logits = student_logits[keep]
            teacher_logits = teacher_logits[keep]

        tau = self.temperature
        student_log_prob = F.log_softmax(student_logits / tau, dim=-1)
        teacher_prob = F.softmax(teacher_logits.detach() / tau, dim=-1)

        loss = F.kl_div(student_log_prob, teacher_prob, reduction="batchmean")
        if self.scale_by_temperature_squared:
            loss = loss * (tau ** 2)
        return loss
