"""Mask-aware objectives for physiology-semantic tokenization."""

from __future__ import annotations

from typing import Dict, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.teachers.physical_state_teacher import PhysicalTeacherOutput
from src.tokenizers.physiology_semantic_tokenizer import ModalityTokenizerOutput


class PhysiologySemanticLoss(nn.Module):
    def __init__(
        self,
        state_weight: float = 1.0,
        prototype_weight: float = 1.0,
        masked_state_weight: float = 1.0,
        reconstruction_weight: float = 1.0,
        vq_weight: float = 1.0,
        private_weight: float = 0.0,
        eeg_coordinate_mask: torch.Tensor | None = None,
        fnirs_coordinate_mask: torch.Tensor | None = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weights = {
            "state": float(state_weight),
            "prototype": float(prototype_weight),
            "masked_state": float(masked_state_weight),
            "reconstruction": float(reconstruction_weight),
            "vq": float(vq_weight),
            "private": float(private_weight),
        }
        self.eps = float(eps)
        self.register_buffer(
            "eeg_coordinate_mask",
            torch.ones(6, dtype=torch.bool) if eeg_coordinate_mask is None else eeg_coordinate_mask.bool(),
        )
        self.register_buffer(
            "fnirs_coordinate_mask",
            torch.ones(9, dtype=torch.bool) if fnirs_coordinate_mask is None else fnirs_coordinate_mask.bool(),
        )
        if self.eeg_coordinate_mask.shape != (6,) or self.fnirs_coordinate_mask.shape != (9,):
            raise ValueError("Coordinate masks must have shapes [6] for EEG and [9] for fNIRS")

    def _masked_uncertainty_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        uncertainty: torch.Tensor,
        mask: torch.Tensor,
        coordinate_mask: torch.Tensor,
    ) -> torch.Tensor:
        if prediction.shape != target.shape or target.shape != uncertainty.shape:
            raise ValueError("Prediction, target, and uncertainty shapes must match")
        if mask.shape != prediction.shape[:2]:
            raise ValueError("Mask must have shape [B,N]")
        if not coordinate_mask.any():
            return prediction.sum() * 0.0
        weighted = (prediction - target).square() / uncertainty.clamp_min(self.eps)
        per_patch = weighted[..., coordinate_mask].mean(dim=-1)
        mask_float = mask.to(per_patch.dtype)
        return (per_patch * mask_float).sum() / mask_float.sum().clamp_min(1.0)

    def _modality_losses(
        self,
        output: ModalityTokenizerOutput,
        target: torch.Tensor,
        uncertainty: torch.Tensor,
        teacher_mask: torch.Tensor,
        coordinate_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        context_mask = teacher_mask & output.context_valid_mask
        return {
            "state": self._masked_uncertainty_loss(
                output.state_prediction, target, uncertainty, teacher_mask, coordinate_mask
            ),
            "prototype": self._masked_uncertainty_loss(
                output.prototype_state_prediction, target, uncertainty, teacher_mask, coordinate_mask
            ),
            "masked_state": self._masked_uncertainty_loss(
                output.context_state_prediction, target, uncertainty, context_mask, coordinate_mask
            ),
            "reconstruction": F.mse_loss(output.reconstruction, output.patches),
            "vq": output.quantizer.commitment_loss,
            "private": output.residual.square().mean() * 0.0,
        }

    def forward(
        self,
        outputs: Mapping[str, ModalityTokenizerOutput],
        teacher: PhysicalTeacherOutput,
    ) -> Dict[str, torch.Tensor]:
        eeg = self._modality_losses(
            outputs["eeg"], teacher.eeg_target, teacher.eeg_uncertainty, teacher.valid_mask,
            self.eeg_coordinate_mask,
        )
        fnirs = self._modality_losses(
            outputs["fnirs"], teacher.fnirs_target, teacher.fnirs_uncertainty, teacher.valid_mask,
            self.fnirs_coordinate_mask,
        )
        components: Dict[str, torch.Tensor] = {}
        total = outputs["eeg"].semantic_latent.sum() * 0.0
        for name, weight in self.weights.items():
            component = eeg[name] + fnirs[name]
            components[name] = component
            total = total + weight * component
        components["total"] = total
        return components


__all__ = ["PhysiologySemanticLoss"]
