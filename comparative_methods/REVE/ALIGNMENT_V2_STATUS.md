# REVE adapter-alignment v2 status

REVE has reached its public-development terminal state in the serial queue.
All six supported classification cells pass A0–A8, REFED remains preregistered
unsupported, and protected evaluation remains separately locked. The active
delivery method has advanced to BrainFusion.

## Retained public evidence

| Task | Unique public samples | Feature shape | Track | Status |
| --- | ---: | ---: | --- | --- |
| Motor imagery | 1,740 | `(1740, 512)` | known target-corpus overlap | A0–A8 pass; protected locked |
| Mental arithmetic | 1,740 | `(1740, 512)` | known target-corpus overlap | A0–A8 pass; protected locked |
| WG | 1,560 | `(1560, 512)` | official pretrained probe | A0–A8 pass; protected locked |
| N-back | 702 | `(702, 512)` | official pretrained probe | A0–A8 pass; protected locked |
| DSR | 8,980 | `(8980, 512)` | official pretrained probe | A0–A8 pass; protected locked |
| Visual | 7,720 | `(7720, 512)` | official pretrained probe | A0–A8 pass; protected locked |
| REFED regression | n/a | n/a | official pretrained probe | unsupported |

All 22,442 supported public sample identities were covered exactly once. Every
one of the 512 feature coordinates is nonconstant in every supported task, and
identical-input replay is bitwise deterministic. The joint schema audit checks
all 13 exact direct-comparison fields against both retained BIOT and CBraMod
peers for all seven comparison groups.

REFED is preregistered unsupported because the frozen snapshot has no temporal
mask argument that could keep padded terminal values out of transformer
attention and query pooling.

## Method-native boundary retained in evidence

The adapter receives the complete shared real-time input support, but official
REVE patching discards an incomplete final patch. At 8 seconds, eight
200-sample patches with 20-sample overlap cover 1,460 of the delivered 1,600
samples; at 2 seconds, two patches cover 380 of 400 samples. This is retained
as a method-native `patch_and_token_grid` field and is not misreported as an
equal effective receptive interval. No padding or copied data is introduced.

The position bank, checkpoint code, checkpoint weights, representation layer,
pooling query, and upstream source revision are all independently identified.
Unlike the upstream REVE LP implementation, the query token is frozen and only
the later A8 linear head will be trainable, preserving the equal-capacity probe
boundary across methods.

## Public-development completion

The A8 runner performs fold-local public feature caching, train-only
standardization and selection, then train-plus-public-validation refitting. Its
weights-only checkpoint and retained predictions are independently audited.
The authorized matrix completed all 90 fold/seed jobs with zero failures, zero
automatic retries, and maximum concurrency one. Every result remains public
development evidence and is explicitly non-table-admissible.

The compact completion evidence is
[`evidence/public_development_v2/matrix_completion_summary.json`](evidence/public_development_v2/matrix_completion_summary.json),
and the terminal cell bundle is
[`evidence/alignment_v2/summary_final.json`](evidence/alignment_v2/summary_final.json).
No protected manifest or array was opened. REVE protected execution is not
authorized and may not overlap BrainFusion, now the sole active delivery method.
