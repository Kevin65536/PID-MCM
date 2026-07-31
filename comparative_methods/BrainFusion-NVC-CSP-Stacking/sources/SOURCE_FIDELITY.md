# BrainFusion NVC-CSP stacking source-fidelity boundary

The official checkout contains TU-Berlin data I/O, modality preprocessing, and
an NVC implementation that convolves processed EEG with an SPM HRF and
correlates it with fNIRS.

The same checkout only names `NVC CSP` and `Integrated Model` in its GUI. Its
`run_analysis` path explicitly simulates completion and does not fit the
paper-case CSP or ensemble. The full case-study execution is therefore not
recoverable from the published source at the pinned revision.

The comparison must independently implement CSP and stacking with every
transform fitted inside the outer-training boundary. Until complete source is
released, the method name is
`brainfusion_nvc_csp_stacking_reimplementation`, and B3 can establish component
fidelity only—not numerical reproduction of the paper's 95.5% within-subject
case result.
