# NormWear adapter-alignment v2 status

NormWear has completed source identity, A0 cell registration, and the full
public A1-A4 data-boundary audit as an explicitly named EEG-fNIRS adaptation.
The executable model gates A5/A7 and protocol freeze A8 remain pending;
protected evaluation is locked.

| Task | Unique public samples | Measured input | Current status |
| --- | ---: | --- | --- |
| Motor imagery | 1,740 | EEG 30; HbO/HbR 36 locations | A0-A4, A6 pass |
| Mental arithmetic | 1,740 | EEG 30; HbO/HbR 36 locations | A0-A4, A6 pass |
| WG | 1,560 | EEG 28; HbO/HbR 36 locations | A0-A4, A6 pass |
| N-back | 702 | EEG 28; HbO/HbR 36 locations | A0-A4, A6 pass |
| DSR | 8,980 | EEG 28; HbO/HbR 36 locations | A0-A4, A6 pass |
| Visual | 7,720 | EEG 30; HbO/HbR 24 locations | A0-A4, A6 pass |
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

Retained public evidence is in `evidence/alignment_v2`. It records A5 and A7 as
pending because no checkpoint forward is part of this phase. This distinction
prevents a validated data loader from being promoted into an executable adapter
claim.
