# B2-B3 test record（STOPPED）

`test_cbramod_smoke.py` covers hash-verified `weights_only=True` loading,
strict official state-dict compatibility, input-contract rejection checks, and
a GPU frozen-encoder probe through forward/backward, optimizer step, and strict
reload.

The former provisional-pooling note is superseded by the completed representation
audit and A0-A8/public evidence. These tests are retained for audit and do not
authorize new work.
