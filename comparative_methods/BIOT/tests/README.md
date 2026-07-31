# B2-B3 test queue

`test_biot_smoke.py` now covers hash-verified `weights_only=True` loading and
strict state-dict compatibility for all three official encoders. It also runs
the frozen encoder plus a linear head on GPU through finite forward/backward,
one optimizer step, and strict checkpoint reload.

B1 channel-panel selection and the source-task numerical comparison remain
pending, so these smoke results do not yet pass B1, B3, or authorize B4.
