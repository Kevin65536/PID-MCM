# B2-B3 test queue

`test_cbramod_smoke.py` covers hash-verified `weights_only=True` loading,
strict official state-dict compatibility, input-contract rejection checks, and
a GPU frozen-encoder probe through forward/backward, optimizer step, and strict
reload.

Mean token pooling is provisional until B3 source-fidelity and the shared EEG
support are frozen. These tests therefore do not authorize B4.
