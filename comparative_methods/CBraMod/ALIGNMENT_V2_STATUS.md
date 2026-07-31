# CBraMod adapter-alignment v2 status

CBraMod is the active serial comparison method. Its representation layer has
been corrected to the official downstream latent-token boundary, and all six
supported classification cells have completed full-public adapter replay with
A0–A7 passing. A8 public-development execution remains pending and is not
self-authorized by this audit. REVE has not started.

| Task | Unique public samples | Representation result | Disposition |
| --- | ---: | --- | --- |
| Motor imagery | 1,740 | all 200 coordinates nonconstant | A0–A7 pass; A8 pending |
| Mental arithmetic | 1,740 | all 200 coordinates nonconstant | A0–A7 pass; A8 pending |
| WG | 1,560 | all 200 coordinates nonconstant | A0–A7 pass; A8 pending |
| N-back | 702 | all 200 coordinates nonconstant | A0–A7 pass; A8 pending |
| DSR | 8,980 | all 200 coordinates nonconstant | A0–A7 pass; A8 pending |
| Visual | 7,720 | all 200 coordinates nonconstant | A0–A7 pass; A8 pending |
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

Both a connectivity smoke and a full MI/outer0/seed17 pilot pass independent
artifact auditing. They share the same retained fold-specific feature cache;
their validation scores remain development-only and are not table-admissible.
The retained 90-job candidate matrix is strictly serial
(`max_concurrent_jobs=1`), permits no automatic retry, and is explicitly not
self-authorizing.

The next serial step is review of the controller followed by a separate public
launch authorization. Neither the candidate matrix nor its pilot authorizes
protected evaluation or concurrent REVE work.
