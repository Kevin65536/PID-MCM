# Alignment configuration（STOPPED）

> **STATUS: STOPPED.** The alignment and 90-job public-development configuration
> below records a completed historical scope; it is not a current launch or
> protected-evaluation authorization.

`alignment_v2.yaml` preregisters the public-only support-matched cells and
freezes the EEG/HbO/HbR, resampling, CWT, representation, and pooling boundary.
It does not authorize protected evaluation. The optional MSiTF route remains
outside this protocol.

`public_development_v2.yaml` freezes the downstream public-only linear-probe
grid, five outer folds, three seeds, zero automatic retries, and a one-job GPU
lane. It consumes only the reviewed A7 task cache and does not authorize the
full matrix or protected evaluation by itself.

`public_job_matrix_candidate_v2.yaml` fixes the 90 serial job identities;
`public_matrix_launch_v2.yaml` authorizes only that reviewed public matrix. The
completed launch did not authorize protected evaluation or concurrent new
comparison-method work.
