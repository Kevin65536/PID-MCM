# BIOT source-fidelity boundary

The code and all three EEG checkpoints come from the official BIOT repository
at the revision in `method_manifest.yaml`. The checkpoint corpus descriptions
are taken from that revision's README.

The planned project route reuses the encoder but replaces upstream task
loaders, splits, and supervised heads with the shared benchmark contract. It is
therefore an official-pretrained transfer evaluation, not a reproduction of
the paper's TUAB/TUEV tables. B3 remains pending until the official example and
the project adapter agree on encoder loading and output semantics.

