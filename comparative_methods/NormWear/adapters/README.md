# Audited adapter

`normwear.py` verifies and loads only the official checkpoint's frozen encoder,
resamples canonical EEG/HbO/HbR to 65 Hz, executes the pinned optimized CWT,
and exposes the official mean-token then channel-concatenation representation.

Per-channel self-attention is executed in configurable chunks to bound GPU
memory. The even-layer cross-channel CLS fusion is still evaluated over the
complete real channel inventory, so chunking does not split the model's channel
interaction. The unchunked path is bitwise identical to upstream; the chunked
path is accepted only under retained floating-point error bounds. Missing
support is rejected rather than padded.
