# BrainFusion NVC-CSP stacking comparison workspace

> **STATUS: STOPPED（legacy method record）。** BrainFusion alignment, public delivery
> and its later joint-campaign cell evaluation are complete. This workspace is retained
> for evidence and replay only; no queue, launch, or protected follow-up here is current.

_Project-level execution and scientific verdicts are generated in
[`docs/PROJECT_STATUS.md`](../../docs/PROJECT_STATUS.md). Public-development and
data-boundary statements here describe the frozen implementation/evidence layer._

This workspace contains the traditional paired EEG-fNIRS fusion baseline.
A0–A8 alignment and the 75-job public-development matrix completed before the later
joint campaign. There is no pretrained
checkpoint: NVC features, CSP filters, base learners, and the stacking learner
must all be fitted inside each outer-training fold.

- Official framework checkout: `upstream/`, revision
  `1d9dcf4026f237efed7f0dd44ba44ef0bf87915b`.
- Public NVC reference:
  `upstream/src/BrainFusion/pipeLine/coupling_analysis.py`.
- Source-fidelity finding and paper reference:
  `sources/method_manifest.yaml`.

The checkout exposes NVC calculation, but the UI's `NVC CSP` and
`Integrated Model` execution is a simulation placeholder. Unless the authors
release the complete paper-case pipeline, project results must be named
`brainfusion_nvc_csp_stacking_reimplementation` and must not be described as a
numerical reproduction of the source paper.

The frozen public-development snapshot is recorded in
`evidence/alignment_v2/summary_final.json` and
`evidence/public_development_v2/matrix_completion_summary.json`. DSR and REFED
are preregistered unsupported; public validation results alone are not
table-admissible.
