# Croce 2017 Real-Data Validation Plan

> Rebased: 2026-05-21
> Status: reset after cleaning off-target local experiments and generated artifacts
> Scope: rebuild the Croce validation path around one local signed neural driver per anchor, then audit whether that inverse problem is stable enough on real data to be trusted.

---

## 1. What Git And The Workspace Actually Say

The committed project progress inside this validation line is still very early.

1. Git history shows one committed milestone on 2026-05-18: the Croce validation workspace and planning documents were created.
2. The previously added real-data exploratory scripts and result folders lived only as ignored local artifacts under `croce_validation/`; they were not part of versioned project progress.
3. On 2026-05-21 those ignored exploratory artifacts were cleaned so they cannot keep biasing design decisions.

Current maintained baseline under `croce_validation/` is now only:

1. [croce_validation/CROCE2017_PAPER_MODEL_REVIEW.md](croce_validation/CROCE2017_PAPER_MODEL_REVIEW.md)
2. [croce_validation/scripts/run_croce2017_paper_simulation.py](croce_validation/scripts/run_croce2017_paper_simulation.py)
3. [croce_validation/scripts/run_local_neighborhood_solver_audit.py](croce_validation/scripts/run_local_neighborhood_solver_audit.py)

This is the correct interpretation of current progress: the project now has a clean paper-faithful simulation baseline and a new local-neighborhood audit entrypoint, but it still does not have a completed held-out real-data decision.

---

## 2. Non-Negotiable Constraints

The following constraints are mandatory and overrule older exploratory approximations.

1. `r(t)` is a **local** signed neural driver, not a global whole-head activity curve.
2. No maintained real-data runner may reconstruct `r(t)` from all EEG or all fNIRS channels at once.
3. No maintained real-data runner may replace `r(t)` with EEG power, envelope, or other unsigned scalar proxies.
4. EEG observations must preserve sign.
5. Forward approximations must keep spatial locality explicit.
6. Signal units, coordinate choices, normalization rules, and polarity conventions must be written into every run manifest.

---

## 3. Two-Step Program

The real-data adoption question is decomposed into two steps.

### Step 1. Forward-Model Compromise Under Missing Physics

We do not have subject-specific EEG lead fields or optical Jacobians. Therefore the real-data path must use a controlled local approximation instead of pretending the full forward model exists.

Required approximation structure:

1. choose one anatomical or sensor-space anchor,
2. choose only spatially neighboring EEG channels around that anchor,
3. choose only spatially neighboring fNIRS channels around that anchor,
4. build a local EEG lead-field surrogate from distance decay plus an explicit sign rule,
5. build local optical surrogates from distance decay for the two fNIRS observation families,
6. store the exact selected channels and weights.

The maintained compromise is:

1. EEG surrogate weights are signed and local,
2. fNIRS surrogate weights are positive and local,
3. hidden dynamics remain Croce's five-state model,
4. real-data inference uses deviation coordinates around baseline for `f`, `HbO`, and `Hb` so zero-centered observations remain meaningful.

### Step 2. Solver Audit On Real Data

Even with a local forward approximation, the inverse problem remains underdetermined. The solver must therefore be audited before any physiological claim is upgraded.

Each anchor-level audit must answer four questions:

1. does the inferred `r(t)` reproduce across seeds,
2. does the result beat a time-shift null,
3. does the result beat a spatial-permutation null,
4. do the recovered state and fNIRS observations exhibit a plausible lag structure.

If these checks fail, the correct action is to demote the model or redesign the approximation. The correct action is not to soften the language while keeping the same workflow.

---

## 4. State And Observation Conventions

For real-data audits the maintained state coordinate system is:

$$
x(t) = [s(t), \Delta f(t), \Delta HbO(t), \Delta Hb(t), r(t)]^T
$$

with

$$
f(t) = 1 + \Delta f(t), \quad HbO(t) = 1 + \Delta HbO(t), \quad Hb(t) = 1 + \Delta Hb(t).
$$

This coordinate change is used for one practical reason: real observations are naturally centered around a local baseline, while Croce's original hemodynamic states are written around baseline value 1. Using deviation coordinates makes zero-mean handling explicit without changing the underlying nonlinear dynamics.

Observation rules:

1. EEG stays signed.
2. Real-data EEG may be resampled to the slower audit grid, but only after sign-preserving low-pass filtering.
3. fNIRS observations are handled as two local observation families.
4. If the input is wavelength-like data, use the paper-style linear HbO/Hb mixing interpretation.
5. If the input is already in chromophore channels, record that this is a chromophore observation approximation rather than the original optical measurement model.

---

## 5. Maintained Scripts

### 5.1 Paper Baseline

[croce_validation/scripts/run_croce2017_paper_simulation.py](croce_validation/scripts/run_croce2017_paper_simulation.py)

Purpose:

1. preserve a paper-faithful single-local-source simulation baseline,
2. keep the five-state dynamics executable,
3. provide a controlled sanity check before touching real data.

### 5.2 Local Solver Audit

[croce_validation/scripts/run_local_neighborhood_solver_audit.py](croce_validation/scripts/run_local_neighborhood_solver_audit.py)

Purpose:

1. audit one local anchor at a time,
2. use only neighboring EEG and fNIRS channels,
3. log units and normalization explicitly,
4. report seed reproducibility, time-shift null, spatial null, and lag diagnostics,
5. support both synthetic audit mode and real-data `.npz` bundle mode.

---

## 6. Real-Data Bundle Contract

The maintained real-data runner intentionally does not depend on earlier project code. It expects one pre-extracted local bundle per anchor.

Required `.npz` fields:

1. `eeg`
2. `eeg_fs_hz`
3. `eeg_positions_mm`
4. `fnirs_positions_mm`
5. `fnirs_fs_hz`
6. one of:
   - `fnirs_690` and `fnirs_830`
   - `fnirs_hbo` and `fnirs_hb`
   - `fnirs_primary` and `fnirs_secondary`

Optional anchor fields:

1. `anchor_position_mm`
2. `eeg_anchor_index`
3. `fnirs_anchor_index`

This contract is narrow on purpose. The goal is to keep the inverse-model audit isolated from the rest of the project until the local formulation is trusted.

---

## 7. Minimum Metrics For Every Anchor Audit

Every maintained run must emit the following metrics:

1. EEG reconstruction correlation and RMSE on the local neighborhood,
2. local fNIRS reconstruction correlation and RMSE for both observation families,
3. mean ESS ratio,
4. posterior uncertainty of `r(t)`,
5. lag from `r(t)` to local fNIRS response,
6. pairwise seed reproducibility of `r(t)`,
7. time-shift null gap,
8. spatial-null gap.

Synthetic mode must additionally report RMSE against the known latent states.

---

## 8. Decision Rule

The local Croce path may move toward held-out subject evaluation only if the following hold for most audited anchors on validation data:

1. median seed-to-seed `r(t)` correlation is at least 0.70,
2. ESS ratio is not collapsing,
3. the time-shift null clearly degrades fit,
4. the spatial null clearly degrades fit,
5. the inferred `r(t)` to fNIRS lag stays inside a physiologically plausible window.

If these conditions fail, the current approximation must remain auxiliary or be replaced.

---

## 9. Immediate Next Actions

1. generate per-anchor `.npz` bundles for the real motor-imagery dataset,
2. run the new local solver audit on validation anchors first,
3. inspect whether sign conventions remain stable across anchors,
4. decide whether the local forward approximation needs to be fixed, tuned, or replaced before any held-out subject sweep.