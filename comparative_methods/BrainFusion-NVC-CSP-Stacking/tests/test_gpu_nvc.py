from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import numpy as np
import pytest
import torch


METHOD_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_ROOT = METHOD_ROOT / "adapters"
if str(ADAPTER_ROOT) not in sys.path:
    sys.path.insert(0, str(ADAPTER_ROOT))

from brainfusion_gpu.nvc import (
    NVCConfig,
    brainfusion_cpu_nvc_reference_avg_raw,
    brainfusion_gpu_nvc_avg_raw,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_gpu_reference_targets_the_pinned_public_nvc_source() -> None:
    source = METHOD_ROOT / "upstream/src/BrainFusion/pipeLine/coupling_analysis.py"
    if not source.is_file():
        pytest.skip("pinned BrainFusion checkout is not available locally")
    assert source.stat().st_size == 11140
    assert _sha256(source) == "4a406a9d3f3f2752cede4ddf29bffe296ac906d7fb3af65c8fd01a67404c459d"
    assert NVCConfig().hrf_oversampling == 1


def test_gpu_avg_raw_nvc_matches_cpu_reference_for_every_channel_pair() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the BrainFusion GPU NVC smoke")
    device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
    rng = np.random.default_rng(20260731)
    eeg_numpy = rng.normal(size=(2, 3, 400))
    fnirs_numpy = rng.normal(size=(2, 4, 20))
    config = NVCConfig()

    eeg = torch.tensor(eeg_numpy, dtype=torch.float64, device=device)
    fnirs = torch.tensor(fnirs_numpy, dtype=torch.float64, device=device)
    convolved, correlations = brainfusion_gpu_nvc_avg_raw(eeg, fnirs, config)
    assert convolved.is_cuda and correlations.is_cuda
    assert convolved.shape == (2, 3, 20)
    assert correlations.shape == (2, 3, 4)
    assert torch.isfinite(correlations).all()

    expected_convolved = np.empty((2, 3, 20), dtype=np.float64)
    expected_correlations = np.empty((2, 3, 4), dtype=np.float64)
    for batch in range(2):
        for eeg_channel in range(3):
            for fnirs_channel in range(4):
                cpu_convolved, cpu_correlation = brainfusion_cpu_nvc_reference_avg_raw(
                    eeg_numpy[batch, eeg_channel],
                    fnirs_numpy[batch, fnirs_channel],
                    config,
                )
                expected_convolved[batch, eeg_channel] = cpu_convolved
                expected_correlations[batch, eeg_channel, fnirs_channel] = cpu_correlation

    np.testing.assert_allclose(convolved.cpu().numpy(), expected_convolved, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        correlations.cpu().numpy(),
        expected_correlations,
        rtol=1e-12,
        atol=1e-12,
    )


def test_gpu_nvc_is_differentiable_and_rejects_cpu_or_constant_inputs() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the BrainFusion GPU NVC smoke")
    device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
    eeg = torch.randn(1, 2, 400, device=device, requires_grad=True)
    fnirs = torch.randn(1, 3, 20, device=device)
    _, correlations = brainfusion_gpu_nvc_avg_raw(eeg, fnirs)
    correlations.square().mean().backward()
    assert eeg.grad is not None
    assert torch.isfinite(eeg.grad).all()

    with pytest.raises(ValueError, match="requires CUDA tensors"):
        brainfusion_gpu_nvc_avg_raw(eeg.detach().cpu(), fnirs.cpu())
    with pytest.raises(ValueError, match="non-constant"):
        brainfusion_gpu_nvc_avg_raw(torch.ones_like(eeg.detach()), fnirs)

    with pytest.raises(ValueError, match="identical EEG-summary and fNIRS grids"):
        brainfusion_gpu_nvc_avg_raw(eeg.detach(), fnirs[..., :-1])
