# CBraMod source-fidelity boundary

The model code is pinned to the official GitHub revision; the checkpoint is
pinned separately to the official Hugging Face revision. The upstream code
expects 200 Hz EEG divided into 200-sample patches.

The project will replace downstream datasets, split construction, and heads
while preserving the pretrained encoder. This is an official-pretrained
transfer evaluation, not a reproduction of the paper's downstream numbers.
B3 remains pending until the upstream quick example and the project adapter
produce compatible encoder behavior.

