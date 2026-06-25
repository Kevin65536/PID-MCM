# Round 1: Broad Exploration — Findings & Agent Positions

**Workflow**: `wf_f7a9a76e-427`  
**Question**: Should the next phase focus on (A) tokenizer-level information interaction design, or (B) downstream token sequence pretraining for physiological coupling discovery?

---

## Final Conclusion

**Paradigm B (downstream discovery) strongly favored over Paradigm A (tokenizer coupling).**

> "The evidence strongly favors paradigm (B) — downstream discovery from discretized token sequences — over paradigm (A) — tokenizer-level coupling. Three confirmed claims from Huh et al. (ICML 2023) establish fundamental VQ-VAE training pathologies (STE gradient gap, mode-seeking codebook collapse, gradient starvation of unselected codes) that collectively explain why coupling constraints fail to guide discrete token assignments toward meaningful cross-modal structure."

### 6 After-Synthesis Findings

| # | Finding | Confidence | Core Evidence |
|---|---------|-----------|---------------|
| 1 | **Coupling loss mathematical inadequacy**: The gradient signal from a soft coupling regularizer is too weak to overcome the STE gradient gap and codebook collapse in standard VQ-VAE training | High | Huh et al. 2023, corroborated by Shekhovtsov 2021, Lancucki et al. 2020 |
| 2 | **Cross-modal auxiliary losses CAN shape VQ token semantics** — but only with _shared codebooks_, soft-assignment paths, and architecture-specific design | High | Liu et al. 2021 (CMCM), Baevski et al. 2020 (wav2vec 2.0), Sadok et al. 2024 (VQ-MAE-AV) |
| 3 | **Information-theoretic ceiling binds discrete representations**: With K=64, log2(K)=6 bits per token is the absolute upper bound on cross-modal MI | High | Lancucki et al. 2020, Information bottleneck theory |
| 4 | **Downstream causal discovery from discrete sequences is feasible**: Hawkes processes, Granger causality, and transfer entropy can recover temporal coupling structure from token sequences | Medium-High | Qiao et al. 2023 (Structural Hawkes), Wu et al. 2024 (ISAHP) |
| 5 | **Continuous→discrete information loss is structural**: CCA drop 0.28→0.12 is consistent with information-theoretic bounds, not solely a training artifact | Medium | Zhao et al. 2026 (dimensional collapse), Lancucki et al. 2020 |
| 6 | **Cross-domain lessons favor independent tokenizer training + downstream alignment**: Speech (HuBERT/wav2vec), video-audio, and image-text all prefer separate tokenizer training with post-hoc alignment | Medium | Baevski et al. 2020, DALL-E (Ramesh et al. 2021) |

### 21 Refuted Claims (Examples)

Claims that were **killed** during adversarial verification include:
- "A shared discrete embedding space automatically induces cross-modal alignment" → REFUTED: CMCM shows alignment requires explicit loss, not automatic
- "Stochastic quantization solves the gradient problem entirely" → REFUTED: SQ-VAE reduces but doesn't eliminate gradient variance in deep networks
- "Larger codebook size K can rescue coupling" → REFUTED: fNIRS effective rank stays at 6-8 regardless of K; dimensional collapse is the binding constraint
- "Transformer attention weights directly reveal causal structure" → REFUTED: Attention ≠ causation without explicit causal regularization
- "Transfer entropy on discrete tokens is sufficient for neurovascular coupling discovery" → REFUTED: TE requires careful binning/permutation testing and can be confounded by shared drive

---

## Agent Positions by Search Angle

### Angle 1: Multi-modal VQ-VAE Coupling Gradients

**Position**: Coupling loss gradients in standard VQ-VAE are MATHEMATICALLY INADEQUATE for guiding discrete token assignments.

**Key claims that survived**:
1. STE copies gradients from decoder input to encoder output with _zero dependence_ on the Voronoi region position → gradient signal carries no information about how to change the encoding
2. Commitment loss is asymmetric and mode-seeking → pushes encoder toward existing codewords rather than moving codewords toward useful representations
3. Only selected codewords receive gradients → gradient starvation of unselected codes

**Key sources**: Huh et al. 2023, Shekhovtsov 2021, Lancucki et al. 2020

**Key refutation**: Architecture mismatch — these analyses assume hard STE with gradient-descent codebook updates, not EMA updates with cosine similarity used in the actual project.

---

### Angle 2: Cross-Modal Tokenizer Interaction Architectures

**Position**: Alternatives to soft coupling regularizers exist, but each has trade-offs for the EEG-fNIRS case.

**Key claims that survived**:
1. Cross-modal attention between encoder latent states provides a differentiable information path bypassing the codebook
2. Contrastive objectives (InfoNCE) between continuous pre-quantization latents can align representations before discretization
3. JEPA-style predictive coding can model cross-modal relationships in latent space

**Key sources**: FCCID (ICLR 2025), VLSA (arXiv 2024), M3-Jepa (arXiv 2024)

---

### Angle 3: Temporal Pattern Discovery from Discrete Event Sequences

**Position**: Downstream causal discovery from discretized token sequences is the MORE PROMISING path.

**Key claims that survived**:
1. Structural Hawkes processes can recover causal graph structure from discrete event sequences without gradient flow through the tokenizer
2. ISAHP (Instance-wise Self-Attentive Hawkes Process) can learn event-type-specific causal influences
3. Transformer-based Hawkes processes can handle long-range temporal dependencies

**Key sources**: Qiao et al. 2023 (IJCAI), Wu et al. 2024, Shou et al. 2023 (CLeaR)

---

### Angle 4: Information Preservation through VQ Bottleneck

**Position**: The CCA drop (0.28→0.12) is largely STRUCTURAL (information-theoretic), not training-related.

**Key claims that survived**:
1. VQ-VAE is a discrete information bottleneck with I(X;Z) ≤ log2(K) bits — absolute upper bound
2. Dimensional collapse (Zhao et al. 2026) means effective capacity is far below theoretical bound
3. Multi-scale quantization (RVQ, product quantization) can increase effective bit depth

**Key sources**: Lancucki et al. 2020, Zhao et al. 2026, Mentzer et al. 2023 (FSQ)

---

### Angle 5: Lessons from Speech/Video/Biomedical Multimodal Tokenization

**Position**: All successful multi-modal discrete token systems use SEPARATE tokenizer training + downstream alignment.

**Key claims that survived**:
1. wav2vec 2.0 shows auxiliary contrastive loss CAN shape discrete token semantics — but on a single modality, not cross-modal coupling
2. DALL-E trains VQ-VAE independently then uses autoregressive transformer for alignment — tokenizer sees NO cross-modal signal
3. VQ-MAE-AV (Sadok et al. 2024) achieves cross-modal alignment through post-tokenization masking + contrastive loss

**Key sources**: Baevski et al. 2020, Ramesh et al. 2021, Sadok et al. 2024
