# Citation Graph: Paper → Agent → Argument Mapping

## How Papers Were Used Across the Two Workflows

Each paper was cited by specific agents within specific verification rounds. Below is the complete mapping.

---

## Core Papers (appearing in BOTH workflows)

### Huh et al. 2023 — "Straightening Out the Straight-Through Estimator"
- **arXiv**: 2305.08842 | **Venue**: ICML 2023
- **Archive PDF**: `papers/huh2023_straight_through.pdf`
- **Cited by**: Round 1 (3 of 4 confirmed claims) + Round 2 (1 confirmed, 2 refuted)
- **Arguments supported**:
  - [CONFIRMED R1+R2] STE introduces gradient gap ∝ quantization error
  - [CONFIRMED R1] Commitment loss is asymmetric and mode-seeking (Bregman divergence)
  - [CONFIRMED R1] Only selected codewords receive gradient → codebook collapse
  - [REFUTED R2] "STE gradient gap is the ROOT CAUSE of coupling loss failure" — FSQ counterevidence weakens this claim
- **Critical limitation**: Analyzes standard VQ-VAE with hard STE and gradient-descent codebook updates, NOT EMA-updated cosine-similarity quantizers with Gumbel-softmax soft assignment

### Liu et al. 2021 — "Cross-Modal Discrete Representation Learning" (CMCM)
- **arXiv**: 2106.05438 | **Venue**: CVPR 2021 / ACL 2022
- **Archive PDF**: `papers/liu2021_cross_modal_discrete.pdf`
- **Cited by**: Round 1 (1 confirmed) + Round 2 (2 confirmed)
- **Arguments supported**:
  - [CONFIRMED R2] CMCM loss operates on soft code-assignment probabilities BEFORE hard quantization → bypasses STE
  - [CONFIRMED R2] Cross-modal auxiliary loss CAN reshape VQ token semantics
  - [LIMITATION] Requires SHARED codebook (project uses separate codebooks per modality)
  - [LIMITATION] Effect size is modest (R@1: 46.0 vs 45.2 baseline)

### Mentzer et al. 2023 — "Finite Scalar Quantization: VQ-VAE Made Simple" (FSQ)
- **arXiv**: 2309.15505 | **Venue**: ICLR 2024
- **Archive PDF**: `papers/mentzer2023_finite_scalar_quant.pdf`
- **Cited by**: Round 2 only (1 confirmed claim)
- **Arguments supported**:
  - [CONFIRMED R2 — DECISIVE EVIDENCE] FSQ uses IDENTICAL STE gradients as VQ yet achieves ~100% codebook utilization where VQ collapses below 50%
  - Proves representation geometry (bounded, low-dimensional, fixed-grid) is the upstream cause; gradient quality is only the downstream mediator

### Shekhovtsov 2021 — "Bias-Variance Tradeoffs in Single-Sample Binary Gradient Estimators"
- **arXiv**: 2110.03549 | **Venue**: GCPR 2021
- **Archive PDF**: `papers/shekhovtsov2021_bias_variance.pdf`
- **Cited by**: Round 1 (supporting) + Round 2 (1 confirmed, then PARTIALLY REFUTED)
- **Arguments supported**:
  - [CONFIRMED R2] Proposition 1+2: NO temperature setting simultaneously gives low bias AND low variance for Gumbel-Softmax
  - [REFUTED R2] "GS gradient vanishes at O(τ^L) for L-layer networks" — analyzes binary Bernoulli VAE step functions, NOT VQ argmin-over-K-vectors → ARCHITECTURE MISMATCH

---

## Round 1 Papers (Tokenizer Architecture + Downstream Methods)

### Zhao et al. 2026 — "Continuous First, Discrete Later: VQ-VAEs Without Dimensional Collapse"
- **arXiv**: 2605.06870
- **Archive PDF**: `papers/zhao2026_continuous_first.pdf`
- **Arguments**: Trained VQ-VAE representations collapse to 1-2% of full rank. AE warm-up restores from 3-5 to 17-19 effective dimensions. Directly explains fNIRS effective rank = 6-8.

### Wu et al. 2024 — "Learning Granger Causality from Instance-wise Self-attentive Hawkes Processes" (ISAHP)
- **arXiv**: 2402.03726
- **Archive PDF**: `papers/wu2024_granger_hawkes.pdf`
- **Arguments**: ISAHP can recover instance-level Granger-causal relationships from discrete event sequences. First neural point process to distinguish synergistic from non-synergistic event pairs.

### Qiao et al. 2023 — "Structural Hawkes Processes for Learning Causal Structure from Discrete-Time Event Sequences"
- **arXiv**: 2305.05986 | **Venue**: IJCAI 2023
- **Archive PDF**: `papers/qiao2023_structural_hawkes.pdf`
- **Arguments**: Structural Hawkes process recovers causal graph from discrete event sequences. Outperforms attention-based methods for event-type causal discovery.

### Baevski et al. 2020 — "wav2vec 2.0"
- **arXiv**: 2006.11477 | **Venue**: NeurIPS 2020
- **Archive PDF**: `papers/baevski2020_wav2vec2.pdf`
- **Arguments**: Auxiliary contrastive loss on Gumbel-softmax quantized latents successfully shapes discrete token semantics. Architecturally closest precedent to the project's setup.

---

## Round 2 Papers (Adversarial Validation)

### Fifty et al. 2024 — "Restructuring Vector Quantization with the Rotation Trick"
- **arXiv**: 2410.06424 | **Venue**: ICLR 2025
- **Archive PDF**: NOT in archive (referenced extensively — 118 mentions across transcripts)
- **Arguments**: STE ignores Voronoi geometry. Rotation trick improves utilization from <2% to >27%. CRITICAL: exact gradients perform WORSE than STE (r-FID 25.4 vs 19.0).

### Vali et al. 2026 — "DiVeQ: Differentiable Vector Quantization Using the Reparameterization Trick"
- **Venue**: ICLR 2026
- **Archive PDF**: NOT in archive (cited 56 times)
- **Arguments**: Independent corroboration of STE gradient pathology from different research group. Eliminates ALL auxiliary losses and trains end-to-end with only reconstruction loss.

### Lu et al. 2026 — "PCA-VAE: Differentiable Subspace Quantization without Codebook Collapse"
- **arXiv**: 2602.18904
- **Archive PDF**: `papers/lu2026_pca_vae.pdf`
- **Arguments**: Abandons VQ entirely for online PCA. Codebook collapse stems from non-stationary encoder updates creating moving target.

### Sadok et al. 2024 — "VQ-MAE-AV: Cross-Modal Discrete Token Alignment"
- **Archive PDF**: NOT in archive (referenced in journal)
- **Arguments**: STRONGEST EXISTENCE PROOF: separate codebooks + joint MAE over discrete tokens + InfoNCE alignment succeeds. Cross-modal loss works on already-discretized tokens.

---

## Citation Density Map

Paper reference frequency across all sub-agent transcripts:

```
arXiv:2410.06424  (Fifty/Rotation Trick)         ████████████████████████████████████████ 118
arXiv:2509.26469  (DiVeQ)                        █████████████████ 56
arXiv:2412.19128  (Semantic Residual)             ██████████████ 46
arXiv:2106.05438  (CMCM/Liu)                     █████████████ 45
arXiv:2605.06870  (Continuous First/Zhao)         █████████████ 42
arXiv:2602.18896  (NSVQ/TransVQ)                 █████████████ 42
arXiv:2309.15505  (FSQ/Mentzer)                  ███████ 24
arXiv:2110.03549  (Shekhovtsov bias-variance)    ██████ 22
arXiv:2305.08842  (Huh STE)                      ██████ 21
arXiv:2402.03726  (ISAHP/Wu)                     █████ 16
arXiv:2305.05986  (Structural Hawkes/Qiao)       ████ 14
arXiv:2006.11477  (wav2vec 2.0/Baevski)          ███ 13
```

---

## Argument-to-Paper Dependencies

### Argument: "Coupling loss is mathematically inadequate" (R1→R2 PARTIALLY OVERTURNED)
- **Primary**: Huh et al. 2023 → Shekhovtsov 2021 → Lancucki et al. 2020
- **Counter-evidence (R2)**: FSQ (Mentzer 2023) — same STE, no collapse; CMCM (Liu 2021) — cross-modal loss works; Rotation Trick (Fifty 2024) — exact gradients worse than STE

### Argument: "Information-theoretic ceiling is the binding constraint" (R2 ELEVATED)
- **Primary**: Lancucki et al. 2020 (I(X;Z) ≤ log2(K)) → Zhao et al. 2026 (dimensional collapse) → arXiv:2604.03191 (compression gap)
- **Supporting**: Murugesan 2016 (EEG-fNIRS MI measurement)

### Argument: "Downstream discovery is the more promising path" (R1+R2 CONSISTENT)
- **Primary**: Qiao et al. 2023 (Structural Hawkes) → Wu et al. 2024 (ISAHP) → Shou et al. 2023 (influence-aware attention)
- **Supporting**: DALL-E (Ramesh 2021) — separate tokenizer training + post-hoc alignment works at scale

### Argument: "Existing analyses have architecture mismatches" (R2 CONFIRMED)
- **Primary**: Shekhovtsov 2021 (binary Bernoulli, not VQ) → Huh 2023 (hard STE, not soft assignment) → Fifty 2024 (hard STE, single modality)
- **Key gap**: NO published work tests NormEMAVectorQuantizer with cosine similarity, Gumbel-softmax, EMA updates, separate codebooks
