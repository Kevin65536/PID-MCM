# Test suite map

The default suite contains software tests and experiment-contract regression
tests. Historical tests below `tests/archive/` are excluded by `pytest.ini`.

`pytest.ini` collects `tests/` only. Method-local suites under
`comparative_methods/<method>/tests/` must be run explicitly when that method changes,
preferably in separate pytest processes. The `sealed_evidence` marker excludes tests
that inspect ignored campaign artifacts from the default suite; run them explicitly
only in a workspace with the matching evidence restored:

```bash
.venv/bin/python -m pytest -q -m sealed_evidence tests/test_protected_campaign_v1.py
```

Ordinary contract tests use small temporary, non-authorizing fixtures and never
create or alter workspace signing records to make the suite pass.

| Area | Representative files | Why retained |
| --- | --- | --- |
| Data and preprocessing | `test_unified_physiology.py`, `test_event_alignment.py`, `test_clean_physiology_cache.py` | Protect dataset, timing, mask, and cache contracts |
| E0–E2 generation | `test_e0_teacher_validity.py`, `test_e1_*`, `test_e2_semantic_evaluation.py` | Preserve interpretation of the completed negative/engineering results |
| R0/R1/R2 | `test_r0p_*`, `test_build_r1*`, `test_qualify_r1p_*`, `test_r2d_*` | Protect preregistration, no-leakage, qualification, and observability contracts |
| Token Atlas | `test_physiological_patch_features.py`, `test_token_physiology*.py`, `test_token_sequence.py` | Reproduce the development-only descriptive analysis |
| Croce/solver | `test_croce_*`, `test_benchmark_solver_optimizations.py` | Preserve physical-model and cache validation |
| Infrastructure | `test_archive_isolation.py`, `test_experiment_logger.py`, `test_run_metrics_comparison.py` | Protect storage, logging, and comparison behavior |
| Unified project state | `test_project_state.py` | Keep execution and scientific verdict separate, render readable evidence links, optionally audit hashes, and detect stale generated views |
| Documentation figures | `test_physiology_semantic_architecture_svg.py`, `test_experiment_plan_svg.py` | Preserve generated-figure provenance; the experiment-plan figure is a frozen snapshot |

The R1-P sealed tests and several real-data audits require local artifacts, so
the default suite deselects them. After restoring the matching evidence, run
the explicit sealed suite above; never alter signed records merely to make it pass.
