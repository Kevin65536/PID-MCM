# BrainFusion observation-budget and loading audit

BrainFusion is admitted only as the independently implemented
`brainfusion_nvc_csp_stacking_reimplementation`. The pinned public checkout
implements scalar EEG–fNIRS NVC, but its `NVC CSP` and `Integrated Model` run
path does not expose the case-study training implementation.

## Shared input boundary

For every supported cell, the adapter consumes exactly the canonical registry
window from time zero: EEG at 200 Hz and paired HbO/HbR at 10 Hz. It may not
read pre-window baseline or post-window hemodynamic context. The 32-second HRF
kernel is a causal method-local transform; convolution is cropped back to the
observed interval and does not authorize extra samples.

The loader freezes the complete real measured channel inventory separately for
each source dataset. The full public record scan found one invariant inventory
per dataset, zero bad channels, and full recorded support on the six
classification tasks:

| Dataset | Public records | EEG | paired HbO/HbR locations |
| --- | ---: | ---: | ---: |
| EEG-fNIRS single trial | 87 | 30 | 36 |
| Simultaneous EEG-NIRS | 26 (25 for DSR) | 28 | 36 |
| Visual cognitive motivation | 54 | 30 | 24 |

The adapter nevertheless validates every sample independently. Missing,
duplicated, bad, non-finite, padded, or analysis-invalid support is rejected;
it is never repaired by copied channels or silent zero padding.

## Task dispositions

Motor imagery, mental arithmetic, WG, N-back, and Visual use the shared
8-second support-matched profile. Motor imagery is labeled as the source-case
corpus reimplementation; the other supported tasks are explicit cross-task
adaptations.

DSR is preregistered unsupported. Its two-second fNIRS input is synchronized
block context rather than an event-native hemodynamic response and is shorter
than the frozen minimum NVC interval. Extending it would fail A2; running only
the modality-specific CSP branches would cease to be the registered NVC-CSP
stacking method.

REFED is preregistered unsupported because the released case method is a
classification stack and the project dataset requires masked continuous
regression with truthful partial terminal support. Neither labels nor padded
time points may be coerced into a classification cell.

## Fold-local boundary

CSP filters, NVC-pair selection, feature standardizers, base-estimator
selection/fits, out-of-fold stacking predictions, and the linear-SVM
meta-estimator must all be fitted using only the current outer-training
partition. No fitted state belongs in the data adapter or reusable raw-feature
cache.
