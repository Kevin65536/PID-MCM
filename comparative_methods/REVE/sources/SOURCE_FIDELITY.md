# REVE source-fidelity boundary

The training and downstream code is pinned to the official GitHub revision.
The position bank is pinned to its official Hugging Face revision. Encoder
weights remain gated by the authors' Responsible Use Agreement and may only be
downloaded by a user who has accepted it.

The project will preserve 200 Hz EEG input and physical coordinate encoding,
but replace upstream downstream-task loaders, splits, and heads. Single-Trial
is not a target-excluded transfer because `Shin2017A` appears in the paper's
exhaustive pretraining list. B3 remains pending until an accepted encoder can
run the official and project minimal examples.

