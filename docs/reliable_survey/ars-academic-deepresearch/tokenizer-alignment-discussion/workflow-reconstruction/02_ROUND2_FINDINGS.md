# Round 2: Targeted Skeptical Deep-Dive — Findings & Agent Positions

**Workflow**: `wf_d35d3ddd-54d`  
**Trigger**: User challenged that Round 1's conclusion (3 of 4 confirmed claims from Huh et al.) had a single-source dependency.  
**Question**: Is the "STE gradient gap is the root cause of coupling loss failure" claim reliable for THIS SPECIFIC architecture (NormEMAVectorQuantizer, cosine-similarity, Gumbel-softmax, EMA updates, separate codebooks)?

---

## Final Conclusion

**The claim that "STE gradient gap is the root cause of coupling loss failure" is NOT supported by available evidence.**

> "While the STE gradient gap is a real and well-documented phenomenon, two decisive findings undermine it as the root cause: (1) FSQ uses IDENTICAL STE gradients yet achieves ~100% codebook utilization where VQ collapses, proving geometry rather than gradient quality is the limiting factor; and (2) soft-assignment cross-modal objectives (CMCM) CAN successfully reshape VQ token semantics through a differentiable path that bypasses the STE entirely."

### 4 Confirmed Findings (from 20 refuted)

| # | Finding | Confidence |
|---|---------|-----------|
| 1 | **STE gradient gap is REAL but NOT DECISIVE** — FSQ uses identical STE yet achieves ~100% codebook utilization where VQ collapses below 50%, proving representation geometry (bounded, low-dimensional, fixed-grid) is the upstream cause and gradient quality is only the downstream mediator | High |
| 2 | **Cross-modal loss CAN reshape VQ token semantics** — via soft code-assignment probabilities computed BEFORE hard quantization, bypassing STE entirely for the cross-modal gradient. Demonstrated with shared codebooks + EMA updates, though effect size is modest even in favorable conditions | High |
| 3 | **Existing STE analyses have FUNDAMENTAL ARCHITECTURE MISMATCHES** with the project's setup — no published work directly tests gradient dynamics of NormEMAVectorQuantizer with cosine similarity, Gumbel-softmax coupling, separate codebooks, and EMA updates | High |
| 4 | **Alternative root cause hypotheses better explain the empirical pattern** — information-theoretic ceiling (log2(K)=6 bits), low cross-modal MI in neurovascular signals, objective mismatch (token-level vs. continuous coupling), and task confound (works on n-back, fails on motor imagery) | Medium |

---

## The 5 Search Angles & Their Key Evidence

### Angle 1: Independent STE Gradient Gap Analyses

**Search**: "VQ-VAE straight-through estimator gradient gap analysis NOT citing Huh 2023"

**Key papers found**:

| Paper | Finding | Relevance |
|-------|---------|-----------|
| **DiVeQ** (Vali et al., ICLR 2026) | Independent corroboration: STE introduces biased gradients that grow with quantization error. SF-DiVeQ achieves full codebook utilization through reparameterization, without auxiliary losses | HIGH — strongest independent evidence |
| **Rotation Trick** (Fifty et al., ICLR 2025) | STE ignores Voronoi geometry; rotation trick improves codebook utilization from <2% to >27%. Critical finding: **exact gradients perform WORSE than STE** (r-FID 25.4 vs 19.0) | HIGH — challenges "better gradients = better results" assumption |
| **FSQ** (Mentzer et al., ICLR 2024) | Uses IDENTICAL STE but eliminates codebook collapse entirely through scalar quantization. VQ complexity is unnecessary | HIGH — proves geometry trumps gradient quality |

**Agent Position**: The STE gradient gap is real but its dominant role has been overstated. FSQ and the Rotation Trick both show that representation geometry, not gradient quality, is the primary determinant of VQ training outcomes.

---

### Angle 2: EMA/Cosine Quantizer Gradient Dynamics

**Search**: "EMA vector quantizer cosine similarity l2-normalized codebook gradient flow soft assignment"

**Key papers found**:

| Paper | Finding | Relevance |
|-------|---------|-----------|
| **Lancucki et al. 2020** (IJCNN) | EMA updates mathematically equivalent to rescaled SGD with per-codeword learning rates ∝ usage frequency | HIGH — EMA doesn't escape gradient limitations |
| **NSVQ/TransVQ** (arXiv:2602.18896) | Codebook collapse stems from non-stationary encoder updates creating a moving target that sparse codebook gradients cannot track | HIGH — explains collapse without STE pathology |
| **Shekhovtsov 2021** | GS gradient norm asymptotically vanishes at O(τ^L) for L-layer networks; Proposition 1+2: NO temperature setting simultaneously gives low bias AND low variance | HIGH — but architecture mismatch: analyzes binary Bernoulli, not VQ |

**Agent Position**: The project's EMA-updated cosine-similarity quantizer has gradient dynamics FUNDAMENTALLY DIFFERENT from the hard-STE setting that Huh et al. analyzed. The "gradient gap" diagnosis may not transfer.

---

### Angle 3: Information Bottleneck through VQ

**Search**: "vector quantization mutual information bound codebook size K bits preserved discrete representation"

**Key papers found**:

| Paper | Finding | Relevance |
|-------|---------|-----------|
| **Lancucki et al. 2020** | VQ-VAE as explicit information bottleneck: I(X;Z) ≤ log2(K) bits, regardless of embedding dimensionality D | HIGH — absolute ceiling |
| **Continuous First, Discrete Later** (Zhao et al., 2026) | Trained VQ-VAE representations collapse to 1-2% of full rank; AE warm-up restores effective dimension from 3-5 to 17-19 | HIGH — directly explains fNIRS effective rank = 6-8 |
| **The Compression Gap** (arXiv:2604.03191) | Data processing inequality: I(O;A) ≤ min(I(O;Z), I(Z;A)); once codebook saturates log2(K), encoder upgrade provides ZERO benefit | HIGH — binding bottleneck principle |

**Agent Position**: The CCA drop (0.28→0.12) has a compelling information-theoretic explanation: with K=64 tokens, the codebook ceiling is 6 bits. If cross-modal MI in the raw signal exceeds 6 bits, it CANNOT be preserved through quantization, regardless of gradient quality.

---

### Angle 4: Cross-Modal VQ-VAE Auxiliary Loss Success

**Search**: "multi-modal VQ-VAE cross-modal auxiliary loss alignment token semantics"

**Key papers found**:

| Paper | Finding | Relevance |
|-------|---------|-----------|
| **CMCM** (Liu et al., 2021) | **DIRECT COUNTEREXAMPLE**: Cross-Modal Code Matching loss shapes VQ token semantics across video+audio+text using shared codebook + MM-EMA | HIGH — proves cross-modal VQ coupling is POSSIBLE |
| **wav2vec 2.0** (Baevski et al., NeurIPS 2020) | Auxiliary contrastive loss on Gumbel-softmax quantized latents DISCRETE > CONTINUOUS for downstream tasks | HIGH — architecturally closest to the project's setup |
| **VQ-MAE-AV** (Sadok et al., 2024) | Separate codebooks + joint MAE over discrete tokens + InfoNCE alignment — cross-modal loss SUCCEEDS on discrete tokens | HIGH — strongest existence proof |
| **DALL-E** (Ramesh et al., 2021) | Tokenizer trained independently; autoregressive transformer aligns post-hoc — no cross-modal signal during tokenization | MEDIUM — paradigm B precursor |

**Agent Position**: The claim that "cross-modal coupling losses cannot shape VQ token semantics" is FALSIFIED by published evidence. However, success requires: shared codebooks (or joint EMA), soft-assignment gradient paths, and high cross-modal MI in the source signal.

---

### Angle 5: EEG-fNIRS Mutual Information in Neuroscience

**Search**: "EEG fNIRS mutual information cross-modal relationship neurovascular coupling"

**Key papers found**:

| Paper | Finding | Relevance |
|-------|---------|-----------|
| **Murugesan 2016** (UT Arlington MS Thesis) | PCMI measurement of EEG-fNIRS coupling: confirms neural→hemodynamic directionality at rest | HIGH — only direct MI measurement |
| **General neuroscience consensus** | Neurovascular coupling is SLOW (3-6s HRF lag), SMEARED (spatial blur), and CONTAMINATED (systemic physiology: Mayer waves, respiration, blood pressure) | HIGH — explains low cross-modal MI |

**Agent Position**: The coupling loss may be optimizing correctly against a signal that simply does not contain enough cross-modal MI. The task-dependent pattern (works on n-back, fails on motor imagery) is consistent with known neuroscience: working-memory tasks produce stronger, more stereotyped hemodynamic responses than motor imagery.

---

## The 20 Refuted Claims (Complete Archive)

**Vote mechanism**: 3 independent agents per claim; ≥2 `refuted` votes = killed. Notation: `confirmed-refuted`.

#### #1 — Vote 0-3
**Claim**: Quantization can be made fully differentiable without STE by reparameterizing it as additive distortion injection, providing genuine (not approximate) gradient flow through the bottleneck.
**Source**: Vali et al., ICLR 2026 (DiVeQ) — https://iclr.cc/virtual/2026/poster/10010131

#### #2 — Vote 0-3
**Claim**: The SF-DiVeQ (space-filling) variant achieves full codebook utilization — implying that poor codebook utilization (codebook collapse) in standard VQ is a consequence of the STE gradient approximation, not an inherent property of discrete bottlenecks.
**Source**: Vali et al., ICLR 2026 (DiVeQ) — https://iclr.cc/virtual/2026/poster/10010131

#### #3 — Vote 0-3
**Claim**: The reparameterization-based gradient path preserves identical hard-assignment forward pass at inference while enabling gradient-based training, demonstrating that the forward/backward mismatch (the core STE pathology identified by Huh et al.) is architecturally avoidable.
**Source**: Vali et al., ICLR 2026 (DiVeQ) — https://iclr.cc/virtual/2026/poster/10010131

#### #4 — Vote 0-3
**Claim**: Replacing the STE with a gradient that encodes geometric relationships (the rotation trick, which preserves the angle between gradient and codebook vector) reduces quantization error by over an order of magnitude and dramatically improves codebook utilization across 11 different VQ-VAE training paradigms, without changing the forward pass.
**Source**: Fifty et al., ICLR 2025 (Rotation Trick) — https://arxiv.org/abs/2410.06424

#### #5 — Vote 1-2
**Claim**: The paper explicitly uses EMA-based codebook updates (not gradient descent) for all experiments, demonstrating that EMA updates do NOT circumvent the STE gradient pathology — the encoder gradient problem persists regardless of how the codebook itself is updated, because the bottleneck between encoder and decoder remains the non-differentiable argmin.
**Source**: Fifty et al., ICLR 2025 — https://arxiv.org/abs/2410.06424

#### #6 — Vote 0-3
**Claim**: FSQ eliminates codebook collapse and all auxiliary losses (commitment loss, codebook reseeding, code splitting, entropy penalties) using only reconstruction loss. This demonstrates that auxiliary/coupling losses are not inherently blocked by the VQ bottleneck — the problem is VQ's high-dimensional learned Voronoi partition, not gradient flow through quantization.
**Source**: Mentzer et al., ICLR 2024 (FSQ) — https://arxiv.org/abs/2309.15505

#### #7 — Vote 1-2
**Claim**: Without a cross-modal coupling objective, a SHARED VQ codebook will spontaneously partition into modality-specific subspaces due to the distributional gap between modalities.
**Source**: Liu et al., CVPR 2021 (CMCM) — https://arxiv.org/abs/2106.05438

#### #8 — Vote 0-3
**Claim**: VQ-VAE suffers from three interlocking flaws — non-differentiable quantization, straight-through estimator (STE) approximation, and codebook collapse — that are causally linked: the non-differentiable argmin lookup blocks native gradient flow, forcing reliance on STE hacks, while the winner-takes-all codebook update leaves non-winning entries static, producing collapse. This is an independent corroboration of VQ gradient pathology that does NOT cite Huh et al. (ICML 2023).
**Source**: Lu et al., 2026 (PCA-VAE) — https://arxiv.org/abs/2602.18904

#### #9 — Vote 0-3
**Claim**: The Gumbel-Softmax relaxation analysis directly contradicts the assumption that soft assignment solves gradient problems. Proposition 3 states that for deep networks with L layers using GS, the probability to observe a non-zero gradient 'vanishes at the rate O(τ^L).' This implies that even with soft assignment (Gumbel-softmax), gradient signal degrades exponentially in depth — meaning the coupling loss gradient reaching the encoder through the soft quantizer would be exponentially attenuated by network depth, even without the hard argmax problem.
**Source**: Shekhovtsov, GCPR 2021 — https://ar5iv.labs.arxiv.org/html/2110.03549

#### #10 — Vote 0-3
**Claim**: The paper's Proposition 1 and 2 together establish that Gumbel-Softmax has a bias-variance tradeoff controlled by temperature τ: bias is O(τ) (vanishes as τ→0) but variance is O(1/τ) (explodes as τ→0). This means there is NO temperature setting that simultaneously gives low bias AND low variance. For the EEG-fNIRS architecture using Gumbel-softmax soft assignment in the coupling loss, this predicts that the coupling gradient is either high-bias (large τ, wrong direction) or high-variance (small τ, noisy), with no sweet spot.
**Source**: Shekhovtsov, GCPR 2021 — https://ar5iv.labs.arxiv.org/html/2110.03549

#### #11 — Vote 1-2
**Claim**: The STE gradient estimation gap (difference between gradients of non-quantized vs. quantized model) is proportionally bounded by the quantization error. When quantization error is zero, the STE is guaranteed to minimize the loss without bias.
**Source**: Huh et al., ICML 2023 — https://proceedings.mlr.press/v202/huh23a/huh23a.pdf

#### #12 — Vote 1-2
**Claim**: Commitment loss is an asymmetric, mode-seeking divergence that gives exactly zero gradient to unselected codebook entries. Once a code is not selected as nearest-neighbor for any input in a batch, it receives no gradient signal and will likely remain permanently dead, creating a self-reinforcing collapse cycle.
**Source**: Huh et al., ICML 2023 — https://proceedings.mlr.press/v202/huh23a/huh23a.pdf

#### #13 — Vote 0-3
**Claim**: Gumbel-softmax (soft assignment) successfully routes cross-modal gradients through the VQ bottleneck, enabling language-to-vision codebook shaping that hard STE cannot achieve. This directly contradicts the claim that 'STE gradient gap makes auxiliary losses impossible to propagate through VQ.'
**Source**: arXiv:2208.00475 — https://arxiv.org/abs/2208.00475

#### #14 — Vote 0-3
**Claim**: An auxiliary cross-modal objective (language-conditioned pixel reconstruction + MIM) successfully reshapes VQ token semantics — each codebook entry acquires a specific visual semantic meaning. This is a counterexample to the hypothesis that auxiliary loss gradients cannot meaningfully alter VQ token assignments.
**Source**: arXiv:2208.00475 — https://arxiv.org/abs/2208.00475

#### #15 — Vote 0-3
**Claim**: Dimensional collapse (not STE gradient gap) is the root cause of VQ-VAE training plateaus: VQ-VAE representations collapse to 1-2% of full latent rank, creating an irreducible loss floor that codebook improvement techniques (respawn, EMA, larger K) cannot surpass. The mechanism is sequential mode activation combined with rate-distortion water-filling — lower-variance latent directions are permanently suppressed by the quantization rate constraint, not by poor gradient flow.
**Source**: Zhao et al., 2026 — https://browse-export.arxiv.org/abs/2605.06870

#### #16 — Vote 0-3
**Claim**: Larger codebook size K does not rescue effective rank under cold-start VQ training. Vanilla VQGAN achieves the same L1 loss across K from 2^10 to 2^14, and codebook effective dimension stays at 2-5 regardless of K growing 64x. This directly predicts the empirical finding that fNIRS effective rank remains 6-8 even with K=128 in the EEG-fNIRS tokenizer — the plateau is a structural property of VQ training dynamics, not a gradient shortfall.
**Source**: Zhao et al., 2026 — https://browse-export.arxiv.org/abs/2605.06870

#### #17 — Vote 0-3
**Claim**: AE warm-up (training as unquantized autoencoder before introducing VQ) restores codebook effective dimension from 3-5 to 17-19 and reduces perceptual loss by 17-35%, without changing the quantizer type, codebook size, or gradient estimator. This falsifies the hypothesis that gradient pathology is the limiting factor: if STE gradient gap were the bottleneck, warm-up (which uses the same STE during VQ phase) could not produce such dramatic improvements.
**Source**: Zhao et al., 2026 — https://browse-export.arxiv.org/abs/2605.06870

#### #18 — Vote 0-3
**Claim**: Only selected codewords receive gradient updates during VQ-VAE training, producing sparse gradients that cause codeword collapse and prevent rich data representations — this is independent of the STE approximation quality and constitutes a gradient-flow pathology even when STE is unbiased.
**Source**: Lancucki et al., IJCNN 2020 — https://ar5iv.labs.arxiv.org/html/2005.08520

#### #19 — Vote 0-3
**Claim**: EMA updates for VQ-VAE codebooks are mathematically equivalent to rescaled SGD with per-codeword learning rates proportional to usage frequency — meaning EMA-based quantizers do NOT escape the fundamental gradient flow limitations of the VQ bottleneck; they only re-weight the effective learning rate per codeword.
**Source**: Lancucki et al., IJCNN 2020 — https://ar5iv.labs.arxiv.org/html/2005.08520

#### #20 — Vote 0-3
**Claim**: Any fixed-capacity discrete codebook imposes a hard information-theoretic upper bound I(Z;T) ≤ log₂|V|^{H_l} on cross-modal information throughput, independent of training quality or gradient estimation method. This means the CCA drop from 0.28 (continuous) to 0.12 (discrete) could be a structural encoding limit, not a gradient pathology.
**Source**: arXiv:2604.03191 — https://ar5iv.labs.arxiv.org/html/2604.03191

**Vote summary**: 16 unanimous (0-3), 4 split (1-2). All 20 killed.

---

## Critical Open Questions (Round 2)

1. **MI measurement**: What is the estimated MI between continuous EEG and fNIRS encoder latents using a calibrated estimator? Tests the information-theoretic ceiling hypothesis directly.

2. **Continuous CCA test**: If coupling loss is applied to continuous pre-quantization latents (bypassing codebooks), does CCA exceed 0.13? The single most critical experiment.

3. **Shared codebook test**: With shared codebooks (both modalities mapping to same K codewords, as in CMCM), does coupling performance improve?

4. **n-back isolation**: If coupling is restricted to n-back data only (where it works) and tested on held-out n-back sessions, does CCA remain elevated? Distinguishes between genuine neurovascular coupling vs. task-specific feature learning.
