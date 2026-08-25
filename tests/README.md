# Test suite map

`pytest.ini` collects the active `tests/` surface only. Superseded architecture
tests are in the local ignored archive and are not importable or default-collected.
Method-local comparison suites must be run explicitly.

The `sealed_evidence` marker excludes checks that require ignored campaign
artifacts. Run those only with the exact evidence restored:

```bash
.venv/bin/python -m pytest -q -m sealed_evidence tests/test_protected_campaign_v1.py
```

| Area | Representative files | Contract |
| --- | --- | --- |
| Data/preprocessing | `test_unified_physiology.py`, `test_event_alignment.py` | dataset, timing, mask, and cache identity |
| Retained E2/T0 | `test_physiology_semantic_*`, `test_token_physiology*.py` | frozen tokenizer and Atlas replay |
| R0/R1/R2 | `test_r0p_*`, `test_build_r1*`, `test_qualify_r1p_*`, `test_r2d_*` | sealed preregistration, no-leakage, and negative-result contracts |
| Croce/solver | `test_croce_*`, `test_benchmark_solver_optimizations.py` | physical-model and cache validation |
| Infrastructure | `test_archive_isolation.py`, `test_project_state.py` | archive boundary and unified state |
| Figures | `test_physiology_semantic_architecture_svg.py`, `test_experiment_plan_svg.py` | source/provenance consistency |

Ordinary tests use temporary, non-authorizing fixtures. Never alter sealed records
or weaken a test to make a clean checkout green.
