# B2-B3 test queue

`test_reve_smoke.py` now verifies the local base encoder and position-bank
artifacts, rejects unknown coordinates and padded support, and runs a
coordinate-aware frozen probe on GPU through forward/backward, optimizer step,
and strict reload.

Dataset-specific coordinate coverage and official-example parity remain B1/B3
work. REVE-large is outside the first-round primary smoke.
