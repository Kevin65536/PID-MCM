# REVE adapter-alignment v2 status

REVE has completed implementation review and the full-public A0–A7 gate. The
A8 runner and one public-only connectivity pilot have passed review. The
candidate 90-job matrix is retained and explicitly authorized only for serial
public execution. Protected evaluation remains locked.

## Retained public evidence

| Task | Unique public samples | Feature shape | Track | Status |
| --- | ---: | ---: | --- | --- |
| Motor imagery | 1,740 | `(1740, 512)` | known target-corpus overlap | A0–A7 pass |
| Mental arithmetic | 1,740 | `(1740, 512)` | known target-corpus overlap | A0–A7 pass |
| WG | 1,560 | `(1560, 512)` | official pretrained probe | A0–A7 pass |
| N-back | 702 | `(702, 512)` | official pretrained probe | A0–A7 pass |
| DSR | 8,980 | `(8980, 512)` | official pretrained probe | A0–A7 pass |
| Visual | 7,720 | `(7720, 512)` | official pretrained probe | A0–A7 pass |
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

## Next serial gate

The A8 pilot uses MI outer fold 0 and seed 17 with a balanced per-class smoke
subset. Its artifact audit recomputes membership, targets, metrics, feature
identity and the weights-only checkpoint, and explicitly marks the result as
non-table-admissible. The candidate 90-job matrix is serial
(`max_concurrent_jobs=1`), has zero automatic retries, and cannot authorize
itself. The separate launch artifact binds all reviewed commits, source files,
config, pilot and matrix identities; it authorizes only this serial public
queue, with protected evaluation and concurrent BrainFusion work still false.
