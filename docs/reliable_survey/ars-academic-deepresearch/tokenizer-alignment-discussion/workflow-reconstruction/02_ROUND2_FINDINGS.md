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

## The 20 Refuted Claims (Key Refutations)

Claims killed during adversarial verification include:

| Refuted Claim | Why Killed |
|---------------|-----------|
| "FSQ eliminates codebook collapse and all VQ pathologies" | FSQ eliminates collapse but introduces scalar quantization artifacts; trade-off, not panacea |
| "EMA updates for VQ-VAE codebooks are mathematically immune to gradient issues" | EMA is equivalent to rescaled SGD; sparse usage still produces sparse effective updates |
| "Larger codebook size K does not rescue effective representation capacity" | Partially refuted — increasing K helps but with diminishing returns; dimensional collapse persists |
| "Dimensional collapse (not STE gradient gap) is the primary pathology" | These are CO-OCCURRING, not competing explanations; the causal direction is unclear |
| "Gumbel-softmax (soft assignment) successfully solves the gradient problem" | GS has its own severe pathologies: exponential gradient vanishing in deep networks, extreme variance at low τ |
| "The CMCM loss gradient reaches the encoder through a fully differentiable path" | True but the path is through softmin probabilities, which have their own gradient quality issues at low temperature |
| "Any fixed-capacity discrete codebook imposes a hard I(Z;T) ≤ log2(K) bound" | True as an upper bound, but the bound is loose — effective capacity is usually FAR below it |
| "AE warm-up (training as unquantized autoencoder first) prevents codebook collapse" | Preliminary result from single 2026 paper, not independently replicated |

---

## Critical Open Questions (Round 2)

1. **MI measurement**: What is the estimated MI between continuous EEG and fNIRS encoder latents using a calibrated estimator? Tests the information-theoretic ceiling hypothesis directly.

2. **Continuous CCA test**: If coupling loss is applied to continuous pre-quantization latents (bypassing codebooks), does CCA exceed 0.13? The single most critical experiment.

3. **Shared codebook test**: With shared codebooks (both modalities mapping to same K codewords, as in CMCM), does coupling performance improve?

4. **n-back isolation**: If coupling is restricted to n-back data only (where it works) and tested on held-out n-back sessions, does CCA remain elevated? Distinguishes between genuine neurovascular coupling vs. task-specific feature learning.
