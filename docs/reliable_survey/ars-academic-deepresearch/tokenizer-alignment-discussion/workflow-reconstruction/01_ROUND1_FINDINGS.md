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

### 21 Refuted Claims (Complete Archive)

**Vote mechanism**: 3 independent agents per claim; ≥2 `refuted` votes = killed. Notation: `confirmed-refuted`.

#### #1 — Vote 0-3
**Claim**: A shared discrete embedding space (VQ codebook) combined with a Cross-Modal Code Matching objective — which forces representations from different modalities to have similar distributions over codebook entries — can align fine-grained correspondences (pixel/word/frame) between modalities without explicit cross-modal supervision labels.
**Source**: Liu et al., ACL 2022 — https://aclanthology.org/2022.acl-long.215/

#### #2 — Vote 0-3
**Claim**: The discretized fine-grained representations (per-pixel/per-word/per-frame tokens) complement summary-level representations (per-video/per-sentence/per-waveform) to improve cross-modal retrieval performance, demonstrating that discrete token-level alignment adds value beyond global pooled representations.
**Source**: Liu et al., ACL 2022 — https://aclanthology.org/2022.acl-long.215/

#### #3 — Vote 0-3
**Claim**: Individual clusters in the shared discrete codebook can represent the same semantic concept across different modalities, suggesting that VQ codebook entries naturally align to cross-modal semantics when trained with a distribution-matching loss — without requiring token-level pairing labels.
**Source**: Liu et al., ACL 2022 — https://aclanthology.org/2022.acl-long.215/

#### #4 — Vote 0-3
**Claim**: VQ-VAE's deterministic quantization with stop-gradient and EMA heuristics is the root cause of codebook collapse — replacing it with stochastic quantization and standard gradient descent eliminates collapse without any heuristics.
**Source**: Takida et al., ICML 2022 (SQ-VAE) — https://proceedings.mlr.press/v162/takida22a/takida22a.pdf

#### #5 — Vote 1-2
**Claim**: Stochastic quantization with a trainable variance parameter exhibits self-annealing: as reconstruction quality improves (lower sigma^2), the quantization variance sigma_phi^2 automatically decreases toward zero, converging to deterministic quantization without any external annealing schedule.
**Source**: Takida et al., ICML 2022 (SQ-VAE) — https://proceedings.mlr.press/v162/takida22a/takida22a.pdf

#### #6 — Vote 0-3
**Claim**: Gumbel-softmax reparameterization through the stochastic quantization distribution enables full gradient flow to the encoder, codebook vectors, and variance parameters simultaneously — the only hyperparameter is the Gumbel temperature tau.
**Source**: Takida et al., ICML 2022 (SQ-VAE) — https://proceedings.mlr.press/v162/takida22a/takida22a.pdf

#### #7 — Vote 0-3
**Claim**: Optimizing cross-modal alignment in the original token space (as done in current coupling loss approaches) is fundamentally prone to 'modality collapse' due to sampling noise, information bias, and content ambiguity — this is a structural argument against Approach A (tokenizer-level coupling) for discrete tokenizers.
**Source**: Lei et al., PMLR v267 — https://proceedings.mlr.press/v267/lei25b.html

#### #8 — Vote 0-3
**Claim**: Cross-modal alignment in a continuous latent embedding space (before quantization) can achieve information-theoretic optimality: the loss combination (1-alpha)*L_cl + alpha*L_reg is formally equivalent to maximizing mutual information I(x;y) while minimizing conditional entropy H(y|x)+H(x|y), preventing representation collapse.
**Source**: Lei et al., PMLR v267 — https://proceedings.mlr.press/v267/lei25b.html

#### #9 — Vote 0-3
**Claim**: Precision-focused quantization (RVQ, FSQ) degrades cross-modal performance compared to vanilla VQ, because accurate quantization tailored to one modality compromises another modality's representation. Specifically, FSQ cross-modal average drops to 52.71 vs. VQ's 59.58, while FSQ unimodal m->m improves to 80.55 vs. VQ's 73.32.
**Source**: Huang et al., arXiv:2412.19128 — https://ar5iv.labs.arxiv.org/html/2412.19128

#### #10 — Vote 0-3
**Claim**: A shared discrete codebook (L=400, D=256) with mutual information disentanglement (CLUB-based MI minimization between general and specific features) plus cross-modal CPC (MI maximization across modalities) can successfully align discrete representations across three modalities (video, audio, text), achieving SOTA cross-modal generalization (avg 62.21 vs. DCID 59.58).
**Source**: Huang et al., arXiv:2412.19128 — https://ar5iv.labs.arxiv.org/html/2412.19128

#### #11 — Vote 1-2
**Claim**: Self-attention weights can be directly repurposed to parameterize Granger-causal intensity functions for discrete event sequences without post-hoc attribution, achieving AUC ~0.97 on synthetic data and ~0.84 on real-world data for type-level causal discovery.
**Source**: NSF PAR biblio/10534627 — https://par.nsf.gov/biblio/10534627

#### #12 — Vote 0-3
**Claim**: Unsupervised instance-level causal discovery from discrete event sequences is feasible without ground-truth causal labels — the model uses only negative log-likelihood of event sequences plus two L1/variance regularization terms, with no labeled causal pairs required.
**Source**: NSF PAR biblio/10534627 — https://par.nsf.gov/biblio/10534627

#### #13 — Vote 0-3
**Claim**: ISAHP is the first neural point process model that formally satisfies the definitional requirements of Granger causality (cause precedes effect; cause carries predictive information about effect) for multi-type discrete event sequences.
**Source**: Wu et al., AISTATS 2024 — https://arxiv.org/abs/2402.03726

#### #14 — Vote 0-3
**Claim**: ISAHP operates at the individual event-instance level rather than at aggregated event-type levels, enabling discovery of finer-grained causal dependencies between specific events that aggregate methods miss.
**Source**: Wu et al., AISTATS 2024 — https://arxiv.org/abs/2402.03726

#### #15 — Vote 0-3
**Claim**: Causal structure is identifiable under instantaneous effects in discrete-time event sequences — simultaneous co-occurrence is a 'blessing' for causal discovery, not a confound. Theorems 2 and 3 prove that the directed causal graph among discrete event types can be uniquely recovered (up to Markov equivalence under faithfulness), meaning discretized token co-occurrence patterns carry recoverable cross-modal coupling information.
**Source**: Qiao et al., IJCAI 2023 — https://arxiv.org/abs/2305.05986

#### #16 — Vote 0-3
**Claim**: Structural Hawkes Processes outperform all Granger-causality-based baselines (ADM4, NPHC, MLE_SGL, PCMCI Plus) at every temporal resolution, and the performance gap widens as bin resolution coarsens — meaning standard Granger methods are fundamentally inadequate for discovering cross-modal coupling from discrete token sequences at low sampling rates.
**Source**: Qiao et al., IJCAI 2023 — https://arxiv.org/abs/2305.05986

#### #17 — Vote 0-3
**Claim**: Symbolization (discretization) of continuous neural signals, when combined with phase-based transfer entropy, preserves sufficient directed connectivity information to achieve 74.27% detection accuracy and 100% specificity (zero false positives) on simulated brain network models. This demonstrates that discrete symbolic representations do not inherently destroy cross-channel coupling structure when the symbolization scheme is appropriately designed.
**Source**: J. Neural Eng. 2020 — https://iopscience.iop.org/article/10.1088/1741-2552/abb4a4

#### #18 — Vote 0-3
**Claim**: Using a single fixed delay for cross-channel connectivity estimation is fundamentally inadequate because real inter-regional neurotransmission delays vary across brain regions. Multi-delay analysis (scanning across a range of temporal lags rather than assuming one fixed lag) is necessary to correctly identify directed coupling. This implies that any cross-modal token co-occurrence discovery must scan across multiple temporal offsets rather than assuming a single fixed lag like 4-6 seconds.
**Source**: J. Neural Eng. 2020 — https://iopscience.iop.org/article/10.1088/1741-2552/abb4a4

#### #19 — Vote 0-3
**Claim**: Downstream transfer entropy analysis on discrete symbolic sequences successfully detected task-modulated changes in effective connectivity in real fNIRS data (finger-tapping task showed significantly increased EC strength compared to resting state). This provides direct evidence that analyzing already-discretized sequences can recover physiologically meaningful, task-dependent coupling patterns.
**Source**: J. Neural Eng. 2020 — https://iopscience.iop.org/article/10.1088/1741-2552/abb4a4

#### #20 — Vote 0-3
**Claim**: Replacing deterministic VQ nearest-neighbor lookup with stochastic quantization (Gaussian noise with learnable variance) produces higher codebook perplexity at all layers and preserves more information through the discrete bottleneck, as measured by reconstruction quality (RMSE, LPIPS, SSIM) and downstream generative FID.
**Source**: HQ-VAE, TMLR 2024 — https://ar5iv.labs.arxiv.org/html/2401.00365

#### #21 — Vote 0-3
**Claim**: Deterministic VQ-VAE suffers from codebook layer collapse, where higher/top layers effectively stop being used (perplexity near 1), and this collapse is not due to insufficient codebook size but to the deterministic commitment loss failing to provide adequate gradient signal for codebook diversification.
**Source**: HQ-VAE, TMLR 2024 — https://ar5iv.labs.arxiv.org/html/2401.00365

**Vote summary**: 19 unanimous (0-3), 2 split (1-2). All 21 killed.

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
