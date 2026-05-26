# Real-Data Validation Plan

> Rebased: 2026-05-26
> Status: redesigned — r(t) has no endogenous dynamics; EEG proposes, fNIRS selects
> Canonical design: [FAST_SLOW_DESIGN.md](FAST_SLOW_DESIGN.md)

---

## 1. Current Baseline

This workspace validates a **modified Croce-style state-space model** on real
EEG-fNIRS data. The core modification: r(t) has no OU dynamics. It is proposed
from EEG at each PF substep and weighted by fNIRS likelihood only.

Committed baseline:
1. [FAST_SLOW_DESIGN.md](FAST_SLOW_DESIGN.md) — canonical design
2. [scripts/run_local_neighborhood_solver_audit.py](scripts/run_local_neighborhood_solver_audit.py) — real-data solver (needs modification per FAST_SLOW_DESIGN.md Section 6)

---

## 2. Non-Negotiable Constraints

1. r(t) is a **local** signed neural driver, not a global whole-head activity curve.
2. No maintained real-data runner may reconstruct r(t) from all EEG or fNIRS channels at once.
3. No maintained real-data runner may replace r(t) with EEG power, envelope, or unsigned scalar proxies.
4. EEG observations must preserve sign.
5. Forward approximations must keep spatial locality explicit.
6. Signal units, coordinate choices, normalization rules, and polarity conventions must be written into every run manifest.
7. r(t) has **no endogenous dynamics** — no OU, no random walk. All temporal structure comes from EEG observations.
8. EEG forward model is deterministic — no σ_eeg parameter in the forward or likelihood.

---

## 3. Source Locations

All neural sources are placed at fNIRS channel positions. Two EEG forward cases:

- **Case A (local):** only EEG channels within 60 mm of the source
- **Case B (whole-brain):** all EEG channels, distance-weighted

Both cases are tested and compared.

---

## 4. Solver Audit Protocol

Each anchor-level audit answers:

1. Does r̂(t) carry the EEG's task-relevant spectral content (alpha 8-13 Hz, beta 13-30 Hz)?
2. Does fNIRS modify r̂(t) — specifically, does the low-frequency (<0.3 Hz) component of r̂ differ from r_eeg = L⁺ y_eeg?
3. Does the modified r̂ produce better fNIRS reconstruction than the EEG-only r_eeg?
4. Does r̂(t) reproduce across random seeds?
5. Does the time-shift null degrade the fit?
6. Do the hemodynamic states exhibit a physiologically plausible lag relative to r̂?

---

## 5. Minimum Metrics Per Anchor

1. EEG source target reconstruction correlation (mean across local channels)
2. fNIRS source target reconstruction correlation (both wavelengths)
3. PSD of r̂(t): alpha/delta power ratio (must be > 0.05, confirming r̂ is not just DC)
4. Low-frequency difference: ||r̂_low - r_eeg_low|| / ||r_eeg_low|| (must be > 0, confirming fNIRS influence)
5. Mean ESS ratio
6. δr̂(t) magnitude relative to r_eeg(t) (should be small: < 20%)
7. Pairwise seed reproducibility of r̂(t)
8. Time-shift null likelihood degradation
9. Hemodynamic lag (peak correlation lag between r̂ envelope and fNIRS)

---

## 6. Decision Rule

The modified Croce path may proceed toward held-out subject evaluation only if:

1. Median seed-to-seed r̂(t) correlation ≥ 0.60
2. EEG reconstruction correlation > 0.5 (dramatic improvement over old ~0.04)
3. Alpha/delta PSD ratio > 0.05
4. ||r̂_low - r_eeg_low|| > 0 (fNIRS is measurably modifying r̂)
5. ESS ratio > 0.3
6. Time-shift null degrades fit

If these fail, the current approximation must be further redesigned.

---

## 7. Implementation Status

| Component | Status |
|-----------|--------|
| `run_croce2017_paper_simulation.py` | Done (paper-faithful reference) |
| `run_local_neighborhood_solver_audit.py` | Needs modification per FAST_SLOW_DESIGN.md §6 |
| Synthetic Phase 1 validation | Not started |
| Real-data Phase 2 audit | Not started |

---

## 8. Reference

- [FAST_SLOW_DESIGN.md](FAST_SLOW_DESIGN.md) — complete mathematical specification, PF algorithm, and experimental plan
