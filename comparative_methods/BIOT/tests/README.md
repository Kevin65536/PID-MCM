# B2-B3 test record（STOPPED）

`test_biot_smoke.py` now covers hash-verified `weights_only=True` loading and
strict state-dict compatibility for all three official encoders. It also runs
the frozen encoder plus a linear head on GPU through finite forward/backward,
one optimizer step, and strict checkpoint reload.

The old B1/B3 pending note is superseded by the later completed A0-A8/public
evidence. These smoke tests are retained for audit and do not authorize new work.
