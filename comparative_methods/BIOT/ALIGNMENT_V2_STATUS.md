# BIOT adapter-alignment v2 status

_Historical public-development chronology ending 2026-07-31. It is not a current
status source; use [`docs/PROJECT_STATUS.md`](../../docs/PROJECT_STATUS.md). Queue and
data-boundary statements below retain their time-local meaning._

At the pre-A8 checkpoint, BIOT was the only active new comparison method. Its six supported
classification cells have completed full-public adapter replay and pass A0–A7.
REFED regression is preregistered as unsupported because the frozen BIOT
adapter cannot preserve its semantically real partial terminal time support.
No protected manifest or array was opened.

| Task | Unique public samples | Pre-A8 disposition |
| --- | ---: | --- |
| Motor imagery | 1,740 | A0–A7 pass; A8 pending |
| Mental arithmetic | 1,740 | A0–A7 pass; A8 pending |
| WG | 1,560 | A0–A7 pass; A8 pending |
| N-back | 702 | A0–A7 pass; A8 pending |
| DSR | 8,980 | A0–A7 pass; A8 pending |
| Visual | 7,720 | A0–A7 pass; A8 pending |
| REFED regression | — | unsupported: `BIOT_NO_PARTIAL_TIME_MASK_CONTRACT` |

The retained machine evidence is in [`evidence/alignment_v2`](evidence/alignment_v2),
and can be regenerated with:

```bash
.venv/bin/python comparative_methods/BIOT/audit_alignment_v2.py
```

In the subsequent update, A8 passed for all six supported cells after the frozen 90-job public matrix
completed with zero failures and zero retries. REFED remains unsupported.
Protected evaluation is still locked and requires a separate decision.

## Public-development implementation

The BIOT-only runner now performs, in order, fold-public feature extraction,
outer-train-only feature standardization and hyperparameter selection, then a
train-plus-public-validation refit with its own refit-fitted standardizer. It
retains selection predictions, failures, a hash-complete feature-cache
identity, and a `weights_only=True` reloadable refit checkpoint. Feature caches
are fold-specific and cannot contain samples outside that fold's public
train/validation membership.

The retained public pilot audit is
[`evidence/public_development_v2/pilot_audit.json`](evidence/public_development_v2/pilot_audit.json).
Both the connectivity smoke and the full MI/outer0/seed17 public pilot pass;
their validation scores remain development-only and are not table-admissible.
The candidate 90-job matrix is serial (`max_concurrent_jobs=1`), has zero
automatic retries, retains failures, and is explicitly not self-authorizing.
The separate reviewed launch manifest authorized only this public matrix and
kept protected evaluation and CBraMod work unauthorized during execution. The
matrix is now complete, so the delivery queue has explicitly advanced to
CBraMod. BIOT has no live job; its deferred protected evaluation remains locked
and may not run concurrently with the active delivery method.
