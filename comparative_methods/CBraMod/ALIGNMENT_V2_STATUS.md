# CBraMod adapter-alignment v2 status

_Historical public-development snapshot. It is not a current status source; use
[`docs/PROJECT_STATUS.md`](../../docs/PROJECT_STATUS.md). Queue and data-boundary
statements below retain their time-local meaning._

CBraMod has reached its public-development terminal state in the serial queue.
Its representation layer is fixed at the official downstream latent-token
boundary, and all six supported classification cells pass A0–A8. REFED remains
preregistered unsupported. Protected evaluation remains separately locked, and
the active delivery method has advanced to REVE.

| Task | Unique public samples | Representation result | Disposition |
| --- | ---: | --- | --- |
| Motor imagery | 1,740 | all 200 coordinates nonconstant | A0–A8 pass; protected locked |
| Mental arithmetic | 1,740 | all 200 coordinates nonconstant | A0–A8 pass; protected locked |
| WG | 1,560 | all 200 coordinates nonconstant | A0–A8 pass; protected locked |
| N-back | 702 | all 200 coordinates nonconstant | A0–A8 pass; protected locked |
| DSR | 8,980 | all 200 coordinates nonconstant | A0–A8 pass; protected locked |
| Visual | 7,720 | all 200 coordinates nonconstant | A0–A8 pass; protected locked |
| REFED regression | — | not dereferenced | unsupported: `CBRAMOD_NO_PARTIAL_TIME_MASK_CONTRACT` |

The six public-complete tasks cover 22,442 unique samples. Every task passed
exact deterministic replay, finite-feature checks, complete-inventory checks,
and real recorded-support checks. No protected manifest or array was opened.

## Direct alignment result

The executable schema audit loads the retained BIOT cell alongside each new
CBraMod cell. For every comparison group, it verifies exact equality of all 13
required method-neutral fields:

- dataset, task, sample inventory, and split identity;
- target schema, target-valid mask, and primary endpoint;
- observation anchor and EEG interval;
- modality and measured-channel identity;
- recorded-support mask; and
- canonical signal-branch identity and fingerprints.

The delivered sample set is therefore aligned through the canonical signal
boundary. Method-native differences remain visible in each cell's
`adapter_identity`: CBraMod uses 200-sample temporal patches, the official
pre-reconstruction 200-dimensional latent representation, and official
channel/patch average pooling.

The upstream raw-array `/100` operation is deliberately not stacked on the
project's canonical recordwise robust-SD input. The rationale and source hashes
are frozen in [`REPRESENTATION_LAYER_AUDIT.md`](REPRESENTATION_LAYER_AUDIT.md).

## Evidence and replay

Retained machine evidence is in [`evidence/alignment_v2`](evidence/alignment_v2).
It can be regenerated, without protected reads, using:

```bash
PYTHONPATH=. .venv/bin/python comparative_methods/CBraMod/audit_alignment_v2.py
```

## Public-development implementation

The CBraMod-only runner now performs fold-specific public feature extraction,
outer-train-only feature standardization and hyperparameter selection, then a
train-plus-public-validation refit with a separately fitted refit
standardizer. It retains public validation predictions, failure state, a
hash-complete feature-cache identity, and a `weights_only=True` reloadable
refit checkpoint. A cache cannot contain samples outside one fold's public
train/validation membership.

Both a connectivity smoke and a full MI/outer0/seed17 pilot passed independent
artifact auditing. They share the same retained fold-specific feature cache;
their validation scores remain development-only and are not table-admissible.
The separately authorized 90-job matrix then completed with 90 passes, zero
failures, zero retries, and maximum concurrency one. Every retained run passed
artifact re-audit. The compact completion evidence is
[`evidence/public_development_v2/matrix_completion_summary.json`](evidence/public_development_v2/matrix_completion_summary.json),
and the terminal cell bundle is
[`evidence/alignment_v2/summary_final.json`](evidence/alignment_v2/summary_final.json).

Public validation aggregates are development-only and remain explicitly
non-table-admissible. CBraMod protected execution is not authorized and may not
overlap REVE, which is now the sole active delivery method.
