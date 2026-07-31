# BrainFusion NVC-CSP stacking comparison workspace

This workspace prepares the traditional paired EEG-fNIRS fusion baseline.
There is no pretrained checkpoint: NVC features, CSP filters, base learners,
and the stacking learner must all be fitted inside each outer-training fold.

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
