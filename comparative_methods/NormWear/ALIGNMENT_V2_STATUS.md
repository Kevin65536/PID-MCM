# NormWear adapter-alignment v2 status

NormWear has completed source identity, A0 cell registration, and the full
public A1-A4 data-boundary audit as an explicitly named EEG-fNIRS adaptation.
The executable model semantics and source-fidelity gates A5/A6 pass. Full
public production replay A7 now also passes. The A8 public-development runner
and its independent artifact auditor are implemented; both the first
n-back/outer0/seed17 connectivity smoke and the corresponding complete public
selection/refit job pass. The remaining folds, matrix freeze, and protected
evaluation remain pending/locked.

| Task | Unique public samples | Measured input | Current status |
| --- | ---: | --- | --- |
| Motor imagery | 1,740 | EEG 30; HbO/HbR 36 locations | A0-A7 pass; A8 pending |
| Mental arithmetic | 1,740 | EEG 30; HbO/HbR 36 locations | A0-A7 pass; A8 pending |
| WG | 1,560 | EEG 28; HbO/HbR 36 locations | A0-A7 pass; A8 pending |
| N-back | 702 | EEG 28; HbO/HbR 36 locations | A0-A7 pass; A8 outer0 pilot pass |
| DSR | 8,980 | EEG 28; HbO/HbR 36 locations | A0-A7 pass; A8 pending |
| Visual | 7,720 | EEG 30; HbO/HbR 24 locations | A0-A7 pass; A8 pending |
| REFED regression | — | partial terminal support | preregistered unsupported |

All 22,442 supported public identities were read exactly once through the
production data view. Every canonical modality tensor was finite and
nonconstant, every recorded and analysis-valid support mask covered the
declared interval, and all EEG/HbO/HbR channel identities came from the frozen
real measured inventory without copy or padding.

For the five tasks shared with the completed BrainFusion multimodal direct
profile, an independent schema audit confirms exact equality of all 13
method-neutral comparison fields. DSR is additionally supported here because
NormWear is a generic signal encoder rather than an HRF/NVC estimator; its two
seconds of fNIRS remain described only as synchronized block context.

Retained public data evidence is in `evidence/alignment_v2`; retained adapter
smoke evidence is in `evidence/adapter_smoke_v2`. The latter verifies safe
encoder-only loading, upstream equivalence, bounded chunked numerics, bitwise
replay on real DSR and 8-second inputs, a trainable linear head, and zero encoder
gradients. The production path subsequently covered all 22,442 unique public
samples and retained a task-specific float32 cache with checkpoint, code,
configuration, batch, channel-order, inventory, split, and data-branch identity.
Every task's first full cache batch replays bitwise exactly.

The A8 runner consumes the retained A7 representation without loading the
encoder again. For each outer fold it materializes labels only for that fold's
public train/validation membership, fits its coordinate standardizer on
outer-train only, selects a linear-probe candidate on public validation, and
refits only on public train plus validation. The n-back connectivity smoke and
complete outer0/seed17 pilot audits are retained under
`evidence/public_development_v2`; both are explicitly public-development
artifacts and are not admissible as final performance evidence.
