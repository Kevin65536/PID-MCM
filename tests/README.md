# Test suite map

The default suite contains software tests and experiment-contract regression
tests. Historical tests below `tests/archive/` are excluded by `pytest.ini`.

| Area | Representative files | Why retained |
| --- | --- | --- |
| Data and preprocessing | `test_unified_physiology.py`, `test_event_alignment.py`, `test_clean_physiology_cache.py` | Protect dataset, timing, mask, and cache contracts |
| E0–E2 generation | `test_e0_teacher_validity.py`, `test_e1_*`, `test_e2_semantic_evaluation.py` | Preserve interpretation of the completed negative/engineering results |
| R0/R1/R2 | `test_r0p_*`, `test_build_r1*`, `test_qualify_r1p_*`, `test_r2d_*` | Protect preregistration, no-leakage, qualification, and observability contracts |
| Token Atlas | `test_physiological_patch_features.py`, `test_token_physiology*.py`, `test_token_sequence.py` | Reproduce the development-only descriptive analysis |
| Croce/solver | `test_croce_*`, `test_benchmark_solver_optimizations.py` | Preserve physical-model and cache validation |
| Infrastructure | `test_archive_isolation.py`, `test_experiment_logger.py`, `test_run_metrics_comparison.py` | Protect storage, logging, and comparison behavior |
| Documentation figures | `test_physiology_semantic_architecture_svg.py`, `test_experiment_plan_svg.py` | Detect status/figure drift |

Collection baseline at the 2026-07-30 consolidation (recheck after edits):

```text
388 tests collected
```

The R1-P sealed tests and several real-data audits require local artifacts.
Their absence in a clean checkout is not permission to alter the sealed source
or silently skip its scientific checks.
