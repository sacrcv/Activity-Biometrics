"""The overall learning objective (Eq. 6).

    L = L_Bio + lambda_1 * L_Ac + lambda_2 * L_KD + lambda_3 * L_Dis

with ``L_Bio = L_ce + L_tri`` (Eq. 2) and all three ``lambda_i`` set to 0.01
("lambda_i, i in [1,2,3] in Eq. 6 is set to 0.01").

This module owns the frozen teacher. Keeping it here rather than inside
``ABNetModel`` has a concrete benefit: the teacher is part of the *objective*,
not the network, so the inference path cannot accidentally depend on
silhouettes, and the teacher's weights never enter ABNet's checkpoints.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from abnet.losses import ActionLoss, CrossEntropyLabelSmooth, DistillationLoss, DistortionLoss
from abnet.losses import TripletLoss
from abnet.utils.logger import get_logger


class ABNetCriterion(nn.Module):
    """Computes every loss term and their weighted sum.

    Args:
        num_identities: identity label-space size.
        num_actions: activity label-space size.
        multilabel: True for Charades-AB, selecting BCE for ``L_Ac``.
        teacher: a trained, frozen :class:`~abnet.models.gaitgl.GaitGL`, or None
            to disable distillation.
        margin: triplet and distortion margin ``m`` (0.3).
        kd_temperature: ``tau`` (4.0).
        activity_weight / distillation_weight / distortion_weight: the three
            ``lambda_i``.
        label_smoothing: epsilon for the identity cross-entropy; 0 gives the
            plain cross-entropy of Eq. 3.
    """

    def __init__(
        self,
        num_identities: int,
        num_actions: int,
        multilabel: bool = False,
        teacher: Optional[nn.Module] = None,
        margin: float = 0.3,
        kd_temperature: float = 4.0,
        activity_weight: float = 0.01,
        distillation_weight: float = 0.01,
        distortion_weight: float = 0.01,
        label_smoothing: float = 0.0,
        normalize_triplet: bool = False,
        normalize_distortion: bool = False,
    ):
        super().__init__()
        self.identity_loss = CrossEntropyLabelSmooth(num_identities, label_smoothing)
        self.triplet_loss = TripletLoss(margin=margin, normalize_feature=normalize_triplet)
        self.action_loss = ActionLoss(num_actions, mode="bce" if multilabel else "ce")
        self.distillation_loss = DistillationLoss(temperature=kd_temperature)
        self.distortion_loss = DistortionLoss(
            margin=margin, normalize_feature=normalize_distortion
        )

        self.activity_weight = float(activity_weight)
        self.distillation_weight = float(distillation_weight)
        self.distortion_weight = float(distortion_weight)
        self.multilabel = multilabel

        # Registered as a submodule so ``.to(device)`` and ``.eval()`` reach it,
        # but every parameter is frozen and excluded from the optimiser.
        self.teacher = teacher
        if teacher is not None:
            teacher.eval()
            for param in teacher.parameters():
                param.requires_grad_(False)
            get_logger().info(
                "distillation enabled: frozen GaitGL teacher with "
                f"{sum(p.numel() for p in teacher.parameters()) / 1e6:.2f}M parameters"
            )
        else:
            get_logger().warning(
                "no teacher provided: L_KD is disabled. This is the paper's "
                "'w/o K/D' ablation, not the full method."
            )

    # ------------------------------------------------------------------
    def train(self, mode: bool = True):
        """Keep the teacher in eval mode even when the criterion is training.

        Without this the teacher's BatchNorm statistics would drift during
        ABNet training, making the distillation target move for reasons that
        have nothing to do with the student.
        """
        super().train(mode)
        if self.teacher is not None:
            self.teacher.eval()
        return self

    @staticmethod
    def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        valid = targets >= 0
        if not valid.any():
            return torch.zeros((), device=logits.device)
        predicted = logits[valid].argmax(dim=-1)
        return (predicted == targets[valid]).float().mean()

    @staticmethod
    def _multilabel_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Top-1 hit rate: is the highest-scoring class among the true labels?"""
        valid = targets.sum(dim=1) > 0
        if not valid.any():
            return torch.zeros((), device=logits.device)
        predicted = logits[valid].argmax(dim=-1)
        hit = targets[valid].gather(1, predicted.unsqueeze(1)).squeeze(1)
        return (hit > 0).float().mean()

    # ------------------------------------------------------------------
    def forward(self, outputs: Dict[str, torch.Tensor], batch: Dict) -> Dict[str, torch.Tensor]:
        """
        Args:
            outputs: the dict returned by :meth:`ABNetModel.forward`.
            batch: the collated batch, supplying ``pid``, ``action`` and
                (optionally) ``silhouette`` / ``has_silhouette``.

        Returns:
            A dict with ``loss`` plus every individual term and two accuracies,
            all as scalar tensors so the metric logger can average them.
        """
        pid = batch["pid"]
        device = outputs["identity_logits"].device
        zero = torch.zeros((), device=device)

        # --- L_Bio (Eq. 2): cross entropy on y_S, triplet on f_bb ---
        loss_ce = self.identity_loss(outputs["identity_logits"], pid)
        loss_tri = self.triplet_loss(outputs["biometrics"], pid)
        loss_bio = loss_ce + loss_tri

        # --- L_Ac (Section 3.2) ---
        loss_ac = self.action_loss(outputs["action_logits"], batch["action"])

        # --- L_KD (Eq. 1) ---
        loss_kd = zero
        if self.teacher is not None and batch.get("silhouette") is not None:
            with torch.no_grad():
                teacher_out = self.teacher(batch["silhouette"])
            loss_kd = self.distillation_loss(
                outputs["identity_logits"],
                teacher_out["logits"].to(outputs["identity_logits"].dtype),
                mask=batch.get("has_silhouette"),
            )

        # --- L_Dis (Eq. 5) ---
        loss_dis = zero
        if "biometrics_distorted" in outputs:
            loss_dis = self.distortion_loss(
                biometrics=outputs["biometrics"],
                biometrics_distorted=outputs["biometrics_distorted"],
                appearance=outputs["appearance"],
                appearance_distorted=outputs["appearance_distorted"],
            )

        total = (
            loss_bio
            + self.activity_weight * loss_ac
            + self.distillation_weight * loss_kd
            + self.distortion_weight * loss_dis
        )

        act_acc = (
            self._multilabel_accuracy(outputs["action_logits"], batch["action"])
            if self.multilabel
            else self._accuracy(outputs["action_logits"], batch["action"])
        )

        return {
            "loss": total,
            "loss_bio": loss_bio.detach(),
            "loss_ce": loss_ce.detach(),
            "loss_tri": loss_tri.detach(),
            "loss_ac": loss_ac.detach(),
            "loss_kd": loss_kd.detach(),
            "loss_dis": loss_dis.detach(),
            "id_acc": self._accuracy(outputs["identity_logits"], pid).detach(),
            "act_acc": act_acc.detach(),
        }

    # ------------------------------------------------------------------
    @classmethod
    def from_config(
        cls,
        cfg,
        num_identities: int,
        num_actions: int,
        multilabel: bool = False,
        teacher: Optional[nn.Module] = None,
    ) -> "ABNetCriterion":
        loss_cfg = cfg.get("loss", {}) or {}
        return cls(
            num_identities=num_identities,
            num_actions=num_actions,
            multilabel=multilabel,
            teacher=teacher,
            margin=loss_cfg.get("margin", 0.3),
            kd_temperature=loss_cfg.get("kd_temperature", 4.0),
            activity_weight=loss_cfg.get("activity_weight", 0.01),
            distillation_weight=loss_cfg.get("distillation_weight", 0.01),
            distortion_weight=loss_cfg.get("distortion_weight", 0.01),
            label_smoothing=loss_cfg.get("label_smoothing", 0.0),
            normalize_triplet=loss_cfg.get("normalize_triplet", False),
            normalize_distortion=loss_cfg.get("normalize_distortion", False),
        )


class TeacherCriterion(nn.Module):
    """Stage-0 objective for the GaitGL teacher: cross entropy + triplet.

    GaitGL's own recipe, which is also what makes its logits a sensible
    distillation target: a classifier trained to discriminate exactly the
    identity set the student must learn.
    """

    def __init__(
        self,
        num_identities: int,
        margin: float = 0.3,
        label_smoothing: float = 0.0,
        triplet_weight: float = 1.0,
    ):
        super().__init__()
        self.identity_loss = CrossEntropyLabelSmooth(num_identities, label_smoothing)
        self.triplet_loss = TripletLoss(margin=margin)
        self.triplet_weight = float(triplet_weight)

    def forward(self, outputs: Dict[str, torch.Tensor], batch: Dict) -> Dict[str, torch.Tensor]:
        pid = batch["pid"]
        logits = outputs["logits"]
        loss_ce = self.identity_loss(logits, pid)

        # Triplet over the flattened part embedding: GaitGL mines triplets
        # per part, and flattening is the cheaper equivalent that keeps the
        # part structure in the distance.
        loss_tri = self.triplet_loss(outputs["triplet_embedding"].flatten(1), pid)
        total = loss_ce + self.triplet_weight * loss_tri

        predicted = logits.argmax(dim=-1)
        return {
            "loss": total,
            "loss_ce": loss_ce.detach(),
            "loss_tri": loss_tri.detach(),
            "id_acc": (predicted == pid).float().mean().detach(),
        }
