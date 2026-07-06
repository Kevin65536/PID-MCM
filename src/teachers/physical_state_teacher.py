"""Patch pooling adapter for the cached Croce physical-state posterior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn as nn


@dataclass
class PhysicalTeacherOutput:
    full_summary: torch.Tensor
    full_uncertainty: torch.Tensor
    eeg_target: torch.Tensor
    eeg_uncertainty: torch.Tensor
    fnirs_target: torch.Tensor
    fnirs_uncertainty: torch.Tensor
    valid_mask: torch.Tensor
    context_valid_mask: torch.Tensor


class PhysicalStateTeacher(nn.Module):
    """Convert sample-rate posterior tensors into detached two-second targets."""

    state_names = ("s", "delta_f", "delta_hbo", "delta_hb", "r")

    def __init__(
        self,
        patch_duration_s: float = 2.0,
        fnirs_sample_rate_hz: float = 10.0,
        eeg_sample_rate_hz: float = 200.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.fnirs_patch_samples = int(round(patch_duration_s * fnirs_sample_rate_hz))
        self.eeg_patch_samples = int(round(patch_duration_s * eeg_sample_rate_hz))
        self.fnirs_sample_rate_hz = float(fnirs_sample_rate_hz)
        self.eeg_sample_rate_hz = float(eeg_sample_rate_hz)
        self.eps = float(eps)
        if self.fnirs_patch_samples <= 1 or self.eeg_patch_samples <= 1:
            raise ValueError("Teacher patches require at least two samples")

    @staticmethod
    def _reshape_patches(value: torch.Tensor, patch_samples: int) -> torch.Tensor:
        if value.dim() != 3:
            raise ValueError(f"Expected [B,T,D], got {tuple(value.shape)}")
        if value.shape[1] % patch_samples != 0:
            raise ValueError(f"Length {value.shape[1]} is not divisible by patch size {patch_samples}")
        return value.reshape(value.shape[0], value.shape[1] // patch_samples, patch_samples, value.shape[2])

    def _summarize(
        self,
        mean: torch.Tensor,
        variance: torch.Tensor,
        sample_rate_hz: float,
        patch_samples: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean_patch = self._reshape_patches(mean, patch_samples)
        var_patch = self._reshape_patches(variance, patch_samples).clamp_min(0.0)
        time = torch.arange(patch_samples, device=mean.device, dtype=mean.dtype) / sample_rate_hz
        centered_time = time - time.mean()
        slope = (mean_patch * centered_time.view(1, 1, -1, 1)).sum(dim=2)
        slope = slope / centered_time.square().sum().clamp_min(self.eps)
        patch_mean = mean_patch.mean(dim=2)
        total_variance = var_patch.mean(dim=2) + mean_patch.var(dim=2, unbiased=False)
        log_variance = total_variance.clamp_min(self.eps).log()
        summary = torch.cat((patch_mean, slope, log_variance), dim=-1)

        mean_uncertainty = var_patch.mean(dim=2) / patch_samples
        slope_weights = centered_time / centered_time.square().sum().clamp_min(self.eps)
        slope_uncertainty = (var_patch * slope_weights.square().view(1, 1, -1, 1)).sum(dim=2)
        logvar_uncertainty = 2.0 / max(patch_samples - 1, 1) * torch.ones_like(total_variance)
        uncertainty = torch.cat((mean_uncertainty, slope_uncertainty, logvar_uncertainty), dim=-1)
        return summary, uncertainty.clamp_min(self.eps)

    def forward(self, teacher: Mapping[str, torch.Tensor]) -> PhysicalTeacherOutput:
        required = {
            "state_mean",
            "state_var",
            "neural_driver_eeg_rate",
            "neural_driver_var_eeg_rate",
            "teacher_valid_mask",
        }
        missing = required.difference(teacher)
        if missing:
            raise KeyError(f"Missing teacher fields: {sorted(missing)}")

        state_mean = teacher["state_mean"].detach()
        state_var = teacher["state_var"].detach()
        driver = teacher["neural_driver_eeg_rate"].detach()
        driver_var = teacher["neural_driver_var_eeg_rate"].detach()
        mask = teacher["teacher_valid_mask"].detach().bool()
        cache_mask = teacher.get("cache_valid_mask", mask).detach().bool()
        causal_mask = teacher.get("causal_valid_mask", torch.ones_like(mask)).detach().bool()
        if state_mean.shape != state_var.shape or state_mean.shape[-1] != len(self.state_names):
            raise ValueError("state_mean/state_var must share shape [B,T,5]")
        if driver.shape != driver_var.shape or driver.shape[-1] != 1:
            raise ValueError("neural driver mean/variance must share shape [B,T,1]")
        if mask.shape != state_mean.shape[:2] or cache_mask.shape != mask.shape or causal_mask.shape != mask.shape:
            raise ValueError("teacher/cache/causal validity masks must have shape [B,T]")

        state_summary, state_uncertainty = self._summarize(
            state_mean, state_var, self.fnirs_sample_rate_hz, self.fnirs_patch_samples
        )
        driver_summary, driver_uncertainty = self._summarize(
            driver, driver_var, self.eeg_sample_rate_hz, self.eeg_patch_samples
        )
        mask_patch = mask.reshape(mask.shape[0], -1, self.fnirs_patch_samples).all(dim=-1)
        cache_mask_patch = cache_mask.reshape(mask.shape[0], -1, self.fnirs_patch_samples).all(dim=-1)
        causal_mask_patch = causal_mask.reshape(mask.shape[0], -1, self.fnirs_patch_samples).all(dim=-1)
        finite_patch = torch.isfinite(state_summary).all(dim=-1) & torch.isfinite(state_uncertainty).all(dim=-1)
        finite_patch &= torch.isfinite(driver_summary).all(dim=-1) & torch.isfinite(driver_uncertainty).all(dim=-1)
        # Local state/prototype supervision needs only a valid physical
        # posterior.  The fixed-history context objective additionally needs
        # the crop-boundary causal history.  `mask_patch` remains part of the
        # context mask for compatibility with caches that already combine the
        # two conditions.
        valid_mask = cache_mask_patch & finite_patch
        context_valid_mask = mask_patch & cache_mask_patch & causal_mask_patch & finite_patch

        # Summary layout is statistic-major: [all means, all slopes, all log variances].
        s_indices = torch.tensor([0, 5, 10], device=state_mean.device)
        fnirs_indices = torch.tensor([1, 2, 3, 6, 7, 8, 11, 12, 13], device=state_mean.device)
        eeg_target = torch.cat((driver_summary, state_summary.index_select(-1, s_indices)), dim=-1)
        eeg_uncertainty = torch.cat((driver_uncertainty, state_uncertainty.index_select(-1, s_indices)), dim=-1)
        fnirs_target = state_summary.index_select(-1, fnirs_indices)
        fnirs_uncertainty = state_uncertainty.index_select(-1, fnirs_indices)

        return PhysicalTeacherOutput(
            full_summary=state_summary.detach(),
            full_uncertainty=state_uncertainty.detach(),
            eeg_target=eeg_target.detach(),
            eeg_uncertainty=eeg_uncertainty.detach(),
            fnirs_target=fnirs_target.detach(),
            fnirs_uncertainty=fnirs_uncertainty.detach(),
            valid_mask=valid_mask.detach(),
            context_valid_mask=context_valid_mask.detach(),
        )


__all__ = ["PhysicalStateTeacher", "PhysicalTeacherOutput"]
