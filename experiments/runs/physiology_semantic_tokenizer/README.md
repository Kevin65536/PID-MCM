# Physiology-semantic result namespace

This root retains only exact evidence needed by current reports, frozen seals, the
R-series replay surface, and the E2 T0 Token Physiology Atlas. The allowlist and
artifact routes are owned by [`../../RESULTS_INDEX.md`](../../RESULTS_INDEX.md).

Superseded smoke, tuning, failed-architecture, staging, and duplicate run trees were
moved to the local Git-ignored pre-forward-implementation archive. Their old
directory names do not define a current suite or status.

Future implementation output belongs at:

```text
experiments/runs/physiology_semantic_tokenizer/<versioned-suite>/<immutable-run>/
```

Each run must carry its resolved config, source/split/cache identities, manifest,
completion state, declared endpoint, and the smallest result package needed to
audit its conclusion. Directory presence never authorizes measured or protected
data access.
