# Phase 2B: Croce et al. 2017 Physical Model Targets

**Date**: 2026-05-13
**Status**: Active

## Motivation

Phase 2A's EEG source target used `EEG_power_envelope` (μV², non-negative), which broke the
additive decomposition `original = source + observation`:
- Power (μV²) and voltage (μV) have different physical dimensions — subtraction is meaningless
- The non-negative envelope forces the observation branch to carry DC offset and zero-crossing structure
- The continuous-session SMC analysis showed raw EEG broadband power → fNIRS correlation is very weak (~0.01)

## Change

Adopt Croce et al. 2017's joint EEG-fNIRS state-space model for target construction:

1. **Shared neural state `s(t)`**: AR(1)-smoothed EEG broadband power, downsampled to fNIRS rate.
   Controlled by `shared_state_alpha` (default 0.90).

2. **fNIRS source target**: `HRF(s(t))`, rescaled to match per-channel fNIRS statistics.
   Unchanged mechanism, but the driver is now the temporally-smoothed shared state
   instead of raw EEG power.

3. **EEG source target**: `signed_rms_carrier` — per-channel RMS amplitude (μV units)
   × sign(smoothed voltage waveform). Temporally smoothed with the same α.
   - Same physical units as EEG (μV, signed)
   - Additive decomposition is physically meaningful
   - Default mode changed from `rms_envelope` to `signed_rms_carrier`

## New Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `shared_state_alpha` | 0.90 | AR(1) smoothing coefficient. Higher = smoother, less fast EEG. |
| `eeg_target_mode` | `signed_rms_carrier` | EEG source target construction mode |

### Alpha Tuning Guide

| α | Half-life @ 10Hz | EEG bands preserved |
|---|-------------------|---------------------|
| 0.80 | 0.45 s | All bands, mild smoothing |
| 0.90 | 0.66 s | α/β/δ/θ power envelopes |
| 0.95 | 1.35 s | δ/θ power envelopes |
| 0.99 | 6.9 s | Only sub-0.1 Hz hemodynamic |
| 0.998 | 34.6 s | Ultra-slow baseline drift only |

## Validation (SMC analysis on real data)

SMC validation confirmed:
- Preprocessing compliance with phase1 `default.yaml` (EEG 0.5–45 Hz, fNIRS 0.2 Hz lowpass)
- Reconstructed signals are time-synchronous with originals (lag-0 peak)
- HRF convolution correctly absorbs the neurovascular delay
- With α=0.998 the state is too slow for task discrimination; α=0.90 provides a better trade-off

## Files Changed

- `src/tokenizers/factorized_labram_vqnsp.py`: Added `_compute_shared_neural_state`, refactored targets
- `experiments/configs/source_observation/phase2a/`: Added `shared_state_alpha` and `eeg_target_mode`
- `docs/ARCHITECTURE.md`: Updated Section 7 with Croce model
- `src/inference/neurovascular_smc.py`: SMC validation module (new)
- `experiments/scripts/run_croce2017_smc_analysis.py`: SMC analysis script (new)
- `experiments/scripts/validate_croce2017_smc.py`: SMC validation script (new)
