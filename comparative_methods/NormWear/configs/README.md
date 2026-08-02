# Alignment configuration

`alignment_v2.yaml` preregisters the public-only support-matched cells and
freezes the EEG/HbO/HbR, resampling, CWT, representation, and pooling boundary.
It does not authorize protected evaluation. The optional MSiTF route remains
outside this protocol.

`public_development_v2.yaml` freezes the downstream public-only linear-probe
grid, five outer folds, three seeds, zero automatic retries, and a one-job GPU
lane. It consumes only the reviewed A7 task cache and does not authorize the
full matrix or protected evaluation by itself.
