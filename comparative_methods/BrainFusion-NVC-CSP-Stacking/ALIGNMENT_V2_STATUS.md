# BrainFusion adapter-alignment v2 status

BrainFusion has completed implementation review and the full-public A0–A7
gate as an explicitly independent NVC-CSP-stacking reimplementation. Five
8-second classification cells are supported; DSR and REFED are preregistered
unsupported. An independently audited, non-table-admissible public smoke pilot
has passed. The full A8 public-development matrix remains pending and protected
evaluation remains locked.

| Task | Unique public samples | Measured input | Status |
| --- | ---: | --- | --- |
| Motor imagery | 1,740 | EEG 30; HbO/HbR 36 locations | A0–A7 pass |
| Mental arithmetic | 1,740 | EEG 30; HbO/HbR 36 locations | A0–A7 pass |
| WG | 1,560 | EEG 28; HbO/HbR 36 locations | A0–A7 pass |
| N-back | 702 | EEG 28; HbO/HbR 36 locations | A0–A7 pass |
| Visual | 7,720 | EEG 30; HbO/HbR 24 locations | A0–A7 pass |
| DSR | — | two-second synchronized context | unsupported |
| REFED regression | — | masked continuous target/partial support | unsupported |

All 13,462 supported public identities were audited exactly once through the
production data view. Every modality input was finite and nonconstant, all
recorded and analysis-valid masks covered the declared interval, and the first
public item in every task produced bitwise-identical dynamic NVC replay. No
channel, modality, time point, or spatial coordinate was copied or padded.

## Observation and method boundary

Supported cells consume exactly the shared canonical 8-second EEG/HbO/HbR
window. The causal 32-second HRF is a method-local convolution kernel cropped
to observed support; it is not permission to read later fNIRS. DSR is excluded
because two seconds is block context rather than a sufficient event-level NVC
interval. REFED is excluded because the source case is a classification stack
and provides no masked sequence-regression or partial-terminal-support
contract.

The public checkout supplies the scalar NVC component but not the paper-case
CSP/ensemble execution. Dynamic NVC is therefore declared as per-time Pearson
contributions whose sum equals the public scalar coefficient. Pair selection,
four CSP transforms, feature standardizers, base learners, and the grouped-OOF
linear-SVM stack are all outer-training-only fitted state. Complete synthetic
checkpoint reload reproduces predictions exactly.

Retained evidence is in [`evidence/alignment_v2`](evidence/alignment_v2) and can
be regenerated without protected reads using:

```bash
PYTHONPATH=. .venv/bin/python comparative_methods/BrainFusion-NVC-CSP-Stacking/audit_alignment_v2.py --device cuda:1
```

The retained public-only smoke pilot independently reproduces fold membership,
targets, macro-F1, checkpoint predictions, and decision scores. It is not a
performance claim. The next serial gate is a full-fold pilot, followed by a
separately authorized 75-job serial matrix. Public validation results remain
development-only and non-table-admissible until the protocol is frozen.
