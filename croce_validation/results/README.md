# Results

Place standardized Croce real-data validation outputs here.

Recommended run layout:

```text
croce_validation/results/<run_name>/
  run_manifest.json
  metrics.json
  subject_level_metrics.csv
  plots/
  design_summary.md
```

Keep each run self-contained so that held-out evaluation, null comparisons, and stability summaries can be reviewed without referring back to `experiments/results/`.