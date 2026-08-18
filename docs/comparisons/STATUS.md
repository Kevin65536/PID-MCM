# Comparison status

Current comparison execution and scientific verdicts are generated in the
[comparison section of the unified project status](../PROJECT_STATUS.md#对比实验).
This compatibility page keeps a short, reader-facing method summary. It is a dated
navigation aid, not a live process dashboard.

## Method summary

| Method / surface | Recorded progress | Table-facing interpretation |
| --- | --- | --- |
| STA-Net strict five-fold | 70/70 trainings | Method-native context reference; keep its estimand separate from support-matched results |
| BIOT | 90/90 jobs; 3 ready-with-note, 3 rejected; REFED unsupported | Use only admitted cells with the alignment note |
| CBraMod | 90/90 jobs; 4 ready-with-note, 2 rejected; REFED unsupported | Use only admitted cells with the alignment note |
| REVE | 90/90 jobs; 3 ready-with-note, 1 rejected, 2 overlap-only (MI/MA); REFED unsupported | Keep overlap cells in their declared overlap track |
| EFRM | 105/105 jobs; 4 ready-with-note, 3 rejected | Use admitted cells with the alignment note |
| NormWear adapted | 90/90 jobs; 5 ready-with-note, 1 rejected; REFED unsupported | Use admitted cells with the alignment note |
| BrainFusion NVC-CSP Stacking | 75/75 jobs; 3 ready-with-note, 2 rejected; DSR/REFED unsupported | Use admitted cells with the alignment note |
| Joint campaign | 540/540 jobs; 42 cells: 22 ready-with-note, 12 rejected, 2 overlap-only, 6 unsupported | Primary source is the tracked 42-cell result report |
| P0 degradation analysis | Exploratory analysis complete | No single confirmatory causal explanation; keep separate from formal tables |

Use the following documents for their stable roles:

- [`PROTOCOL.md`](PROTOCOL.md): comparison design and evaluation rules;
- [`METRIC_ACCEPTANCE.md`](METRIC_ACCEPTANCE.md): cell-level interpretation policy;
- [`PROTECTED_CAMPAIGN_RESULTS_20260814.md`](PROTECTED_CAMPAIGN_RESULTS_20260814.md):
  dated campaign result snapshot;
- [`PERFORMANCE_DEGRADATION_P0_RESULTS_20260816.md`](PERFORMANCE_DEGRADATION_P0_RESULTS_20260816.md):
  exploratory post-campaign analysis.

For a machine-readable current projection:

```bash
.venv/bin/python experiments/scripts/project_state.py show --format agent
```
