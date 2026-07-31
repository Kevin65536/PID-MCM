"""GPU NVC kernel matched to BrainFusion's published CPU calculation."""

from __future__ import annotations

from dataclasses import dataclass

from nilearn.glm.first_level import spm_hrf
import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class NVCConfig:
    eeg_sampling_rate_hz: int = 200
    fnirs_sampling_rate_hz: int = 10
    eeg_window_samples: int = 20
    hrf_tr: float = 1.0
    hrf_oversampling: int = 16
    hrf_time_length: float = 32.0


def _numpy_minmax(values: np.ndarray) -> np.ndarray:
    value_range = np.max(values) - np.min(values)
    if not np.isfinite(value_range) or value_range <= 0:
        raise ValueError("BrainFusion NVC requires a finite non-constant signal")
    return (values - np.min(values)) / value_range


def brainfusion_cpu_nvc_reference_avg_raw(
    eeg_signal: np.ndarray,
    fnirs_signal: np.ndarray,
    config: NVCConfig = NVCConfig(),
) -> tuple[np.ndarray, float]:
    """Small executable reference of the upstream ``avg_raw`` NVC branch."""

    eeg = np.asarray(eeg_signal, dtype=np.float64)
    fnirs = np.asarray(fnirs_signal, dtype=np.float64)
    window = int(config.eeg_window_samples)
    processed_eeg = np.asarray(
        [np.mean(eeg[index : index + window]) for index in range(0, len(eeg) - window + 1, window)],
        dtype=np.float64,
    )
    processed_eeg = _numpy_minmax(processed_eeg)
    hrf = spm_hrf(
        config.hrf_tr,
        oversampling=config.hrf_oversampling,
        time_length=config.hrf_time_length,
    )
    convolved = np.convolve(processed_eeg, hrf, mode="full")[: len(processed_eeg)]
    convolved_normalized = _numpy_minmax(convolved)
    fnirs_normalized = _numpy_minmax(fnirs)
    length = min(len(convolved_normalized), len(fnirs_normalized))
    correlation = float(
        np.corrcoef(convolved_normalized[:length], fnirs_normalized[:length])[0, 1]
    )
    if not np.isfinite(correlation):
        raise ValueError("BrainFusion CPU reference produced a non-finite correlation")
    return convolved, correlation


def _tensor_minmax(values: torch.Tensor) -> torch.Tensor:
    minimum = values.amin(dim=-1, keepdim=True)
    maximum = values.amax(dim=-1, keepdim=True)
    value_range = maximum - minimum
    if not bool(torch.isfinite(value_range).all()) or bool((value_range <= 0).any()):
        raise ValueError("BrainFusion GPU NVC requires finite non-constant signals")
    return (values - minimum) / value_range


def brainfusion_gpu_nvc_avg_raw(
    eeg: torch.Tensor,
    fnirs: torch.Tensor,
    config: NVCConfig = NVCConfig(),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized CUDA implementation of all EEG-to-fNIRS NVC pairs.

    Parameters
    ----------
    eeg:
        Tensor with shape ``[batch, eeg_channels, eeg_samples]``.
    fnirs:
        Tensor with shape ``[batch, fnirs_channels, fnirs_samples]``.

    Returns
    -------
    convolved_eeg:
        HRF-convolved EEG summaries with shape ``[batch, eeg_channels, time]``.
    correlations:
        Pearson NVC coefficients with shape
        ``[batch, eeg_channels, fnirs_channels]``.
    """

    if eeg.ndim != 3 or fnirs.ndim != 3:
        raise ValueError("BrainFusion NVC inputs must have shapes [B,C,T]")
    if eeg.shape[0] != fnirs.shape[0]:
        raise ValueError("BrainFusion NVC modalities must share a batch size")
    if not eeg.is_cuda or not fnirs.is_cuda:
        raise ValueError("BrainFusion GPU NVC requires CUDA tensors")
    if eeg.device != fnirs.device:
        raise ValueError("BrainFusion NVC modalities must share a CUDA device")
    if not torch.is_floating_point(eeg) or not torch.is_floating_point(fnirs):
        raise TypeError("BrainFusion NVC inputs must be floating-point tensors")
    if not bool(torch.isfinite(eeg).all()) or not bool(torch.isfinite(fnirs).all()):
        raise ValueError("BrainFusion NVC inputs contain non-finite values")
    if config.eeg_sampling_rate_hz // config.fnirs_sampling_rate_hz != config.eeg_window_samples:
        raise ValueError("GPU avg_raw window must preserve the configured EEG/fNIRS rate ratio")

    window = int(config.eeg_window_samples)
    complete_samples = (eeg.shape[-1] // window) * window
    if complete_samples < window:
        raise ValueError("BrainFusion NVC EEG input is shorter than one averaging window")
    processed_eeg = eeg[..., :complete_samples].reshape(*eeg.shape[:-1], -1, window).mean(dim=-1)
    processed_eeg = _tensor_minmax(processed_eeg)

    hrf_numpy = spm_hrf(
        config.hrf_tr,
        oversampling=config.hrf_oversampling,
        time_length=config.hrf_time_length,
    )
    hrf = torch.as_tensor(hrf_numpy, dtype=eeg.dtype, device=eeg.device)
    flattened = processed_eeg.reshape(-1, 1, processed_eeg.shape[-1])
    convolved = F.conv1d(
        F.pad(flattened, (hrf.numel() - 1, 0)),
        hrf.flip(0).reshape(1, 1, -1),
    )
    convolved = convolved.reshape(*processed_eeg.shape)

    normalized_convolved = _tensor_minmax(convolved)
    normalized_fnirs = _tensor_minmax(fnirs)
    length = min(normalized_convolved.shape[-1], normalized_fnirs.shape[-1])
    normalized_convolved = normalized_convolved[..., :length]
    normalized_fnirs = normalized_fnirs[..., :length]
    centered_eeg = normalized_convolved - normalized_convolved.mean(dim=-1, keepdim=True)
    centered_fnirs = normalized_fnirs - normalized_fnirs.mean(dim=-1, keepdim=True)
    numerator = torch.einsum("bet,bft->bef", centered_eeg, centered_fnirs)
    eeg_norm = torch.linalg.vector_norm(centered_eeg, dim=-1)
    fnirs_norm = torch.linalg.vector_norm(centered_fnirs, dim=-1)
    denominator = eeg_norm.unsqueeze(-1) * fnirs_norm.unsqueeze(-2)
    if bool((denominator <= 0).any()):
        raise ValueError("BrainFusion NVC correlation received a constant processed signal")
    correlations = numerator / denominator
    if not bool(torch.isfinite(correlations).all()):
        raise RuntimeError("BrainFusion GPU NVC produced non-finite correlations")
    return convolved, correlations
