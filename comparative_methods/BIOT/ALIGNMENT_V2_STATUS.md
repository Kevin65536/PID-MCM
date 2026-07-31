# BIOT adapter-alignment v2 status

BIOT remains the only active new comparison method. Its six supported
classification cells have completed full-public adapter replay and pass A0–A7.
REFED regression is preregistered as unsupported because the frozen BIOT
adapter cannot preserve its semantically real partial terminal time support.
No protected manifest or array was opened.

| Task | Unique public samples | Disposition |
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

A8 is deliberately still pending. It requires a frozen public-development job
matrix, retry/failure rules, and a separate protected unlock decision. The
serial queue therefore remains on BIOT; CBraMod implementation or execution
must not start yet.

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
Therefore A8 and the BIOT delivery queue remain open pending matrix review and
execution; no protected command is present.
