# B2-B3 test record（STOPPED）

`test_reve_smoke.py` now verifies the local base encoder and position-bank
artifacts, rejects unknown coordinates and padded support, and runs a
coordinate-aware frozen probe on GPU through forward/backward, optimizer step,
and strict reload.

The former dataset-coverage and official-parity pending note is superseded by the
completed A0-A8/public evidence. REVE-large remains outside the stopped primary
scope; these tests do not authorize new work.
