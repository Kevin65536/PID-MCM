# Theoretical foundations of physiology-semantic tokenization

_Formal assumptions, information paths, alignment mechanism, and claim limits_

---

## 📋 Purpose

This document explains why the target representation can retain task-relevant information and why independently generated EEG and fNIRS tokens can exhibit stable delayed correspondence. It also states what cannot be guaranteed by architecture alone.

The theory is deliberately conditional. A tokenizer cannot preserve physiological information that is absent from the measurements, and no unsupervised objective guarantees that finite codebook capacity will be allocated to a desired task variable.

## 🧠 Generative assumptions

Let `Y` denote a task or physiological condition. Let `R_t` and `S_t` denote fast neural and vasoactive states, `H_t` denote hemodynamic states, and `U_t^E`, `U_t^F` denote modality-private physiology and nuisance.

\[
Y\rightarrow \{R_t,S_t,H_t,U_t^E,U_t^F\}_{t=1}^{T}
\rightarrow \{X_t^E,X_t^F\}_{t=1}^{T}
\]

The observations satisfy:

\[
X_t^E=g_E(R_t,S_t,U_t^E)+\epsilon_t^E
\]

\[
X_t^F=g_F(H_t,U_t^F)+\epsilon_t^F
\]

with delayed dynamics:

\[
p(H_t\mid H_{t-1},S_{\le t})
\]

The Croce-style state vector used by the current cache solver is:

\[
x_t=(s_t,\delta f_t,\delta HbO_t,\delta Hb_t,r_t)
\]

The target architecture does not require this five-state model, or any single shared neural driver, as an input or semantic truth. Croce is one testable auxiliary hypothesis among self-supervised, task, dynamical and other physics-informed targets. Every target family must establish its own identifiability and uncertainty scope before it can supervise tokens.[^1]

## 🎯 How task information reaches tokens

### Data-processing limit

For a deterministic or stochastic tokenizer `K = Q(E(X))`, the data-processing inequality gives:

\[
I(Y;K)\le I(Y;X)
\]

Discretization cannot create task information. The design objective is therefore to allocate finite token capacity to task-relevant physiological variation while placing high-fidelity but weakly interpretable detail in the residual stream.

### Representation decomposition

The exported representation is:

\[
T_t=(K_t,Q_t,E[K_t],R_t)
\]

where:

- `K_t` is a nominal physiological state identifier;
- `Q_t` is the posterior over codewords;
- `E[K_t]` is the codebook prototype or posterior-weighted expected embedding;
- `R_t` is the private/residual representation.

The information paths are complementary:

| Representation element | Information retained | Primary role |
| --- | --- | --- |
| Hard ID | Learned measurement-region membership | Counting, transition statistics, interpretability after probing |
| Posterior | Assignment uncertainty and secondary prototypes | Robust coupling and downstream inference |
| Prototype embedding | Geometry among learned regions | Transfer to sequence models |
| Residual latent | Model-private physiology and reconstruction detail | Information preservation |
| Contextual sequence state | Duration, transition, and spatial grammar | Fine-grained task prediction |

### Sufficient-state condition

Assume the task decision depends on a physiological trajectory `S_1:T` through a score function `g`. If `g` is `L`-Lipschitz and the correct class has margin `gamma`, then the decision is unchanged whenever:

\[
\|\hat S_{1:T}-S_{1:T}\| < \frac{\gamma}{2L}
\]

This is not a guarantee that the tokenizer will meet the bound. It identifies the correct empirical question: compare registered signature/probe and task-decision error against quantization distortion rather than relying on signal reconstruction alone.

### Why sequence context is required

Fine-grained cognitive state is unlikely to be a function of one 2-second state ID. It may depend on token dwell time, transition rates, spatial coordination, and delayed cross-modal response. Therefore:

\[
p(Y\mid K_t)\neq p(Y\mid K_{1:T})
\]

in general. The target makes the token sequence, not an isolated token, the downstream unit. This follows the successful pattern of masked contextual representation learning in EEG and general multimodal self-supervision, while retaining a discrete analysis interface.[^2][^3]

## 🗣️ What the language-model analogy transfers

Modern language modeling does not perform reasoning on integer IDs directly. The full path is:

| Stage | Language model | Physiological counterpart | Information status |
| --- | --- | --- | --- |
| Segmentation | Text → subword units | Signal → channel/space/time patches | Defines boundaries; can already destroy event structure |
| Symbolization | Subword → token ID | Encoder patch → codebook ID | ID is only a nominal address |
| Input embedding | ID → learned vector | ID → saved codebook vector or learned embedding | Carries trainable geometry before context |
| Context model | Embedding sequence → hidden states | Multimodal temporal/spatial backbone | Adds sequence, lag, duration, and neighborhood meaning |
| Self-supervision | Next/masked-token likelihood | Masked state/token or future-distribution prediction | Allocates geometry to predictable structure |
| Task adaptation | Prompt/fine-tune/readout | Frozen probe or fine-tuning | Tests whether the representation exposes task information |

Three consequences matter for this project.

First, token IDs never contain linear semantics by themselves. Word2vec-style relations are properties of a learned vector space and its objective, not of the integer vocabulary labels. Arbitrarily permuting IDs leaves the symbolic sequence unchanged if the embedding lookup is permuted with it.

Second, an LLM may learn useful contextual geometry even when its input IDs are nominal, because the embedding table and Transformer are optimized jointly by sequence prediction. Our current downstream path breaks the stronger transfer interpretation when it replaces tokenizer prototypes with fresh `nn.Embedding` parameters: it retains category and transition identity, but initially discards the tokenizer codebook geometry.

Third, a biosignal tokenizer cannot assume that word-like units already exist. The patch boundary, encoder objective, quantizer, and contextual objective jointly determine what becomes a symbol. Therefore the correct analogy is not “physiological token ID equals word”; it is “a learned physiological symbol plus its prototype and context state can play the interface role that a subword token plays in a language model.”

## 🧪 Why reconstruction is necessary but insufficient

Reconstruction and semantic learning answer different questions:

| Objective | What it rewards | What it does not guarantee |
| --- | --- | --- |
| Raw/spectral reconstruction | High-fidelity local signal content | Task relevance, physical state identity, cross-subject stability |
| Physical-state/prototype supervision | Codeword organization by teacher-defined state | Retention of all private/task information |
| Masked state/token prediction | Predictive temporal and spatial grammar | Correct physiological interpretation without grounded targets |
| Cross-modal sequence prediction | Statistical conditional structure | Causal neurovascular coupling or freedom from marginals |
| Supervised task loss | Label utility on the selected distribution | General physiology or transfer to unseen tasks |

Raw reconstruction therefore remains an auxiliary information-preservation objective, not the definition of semantic success. The target architecture combines state/prototype supervision with masked contextual learning and a residual path because no single objective spans interpretability, fidelity, and task utility.

## 🧩 Relation to LaBraM and NeuroRVQ

LaBraM trains a vector-quantized neural-spectrum tokenizer on EEG channel patches, freezes it, and pretrains a Transformer to predict masked neural codes.[^4] NeuroRVQ targets a different bottleneck: multi-scale feature extraction, hierarchical residual VQ, and phase/amplitude-aware reconstruction improve high-frequency fidelity and compression before masked generative modeling.[^5]

| Design axis | LaBraM | NeuroRVQ | Approved EEG–fNIRS target |
| --- | --- | --- | --- |
| Primary modality | EEG | EEG, with broader biosignal motivation | Separate EEG and paired-optical fNIRS |
| Temporal resolution | Primarily fixed channel patches | Explicit multi-scale patches | Initially fixed 2-second grid; multi-scale is a later ablation |
| Quantization | Single VQ neural codes | Hierarchical residual VQ | One semantic VQ plus continuous private residual first |
| Tokenizer target | Neural-spectrum prediction | Phase/amplitude-aware high-fidelity reconstruction | Uncertainty-aware physical state and prototype semantics plus reconstruction |
| Context objective | Masked neural-code prediction | Generative masked token modeling | Masked physical state/token context and frozen cross-modal distribution prediction |
| Cross-modal claim | Not an EEG–fNIRS correspondence model | Multimodal integration is a motivation, not demonstrated EEG–fNIRS coupling | Explicitly tests EEG-history incremental prediction of fNIRS distributions |
| Primary success criterion | Transfer across EEG tasks | Reconstruction/generation and downstream EEG performance | State semantics, information retention, controlled coupling, then downstream utility |

The redesign adopts their separation between tokenizer training and contextual pretraining, but changes the semantic target. It postpones RVQ because adding multiple residual codebooks before the physical-state branch is validated would confound two failures: inadequate semantic organization and insufficient bitrate. NeuroRVQ-style multi-scale RVQ remains a justified E3/E4 extension if the continuous-residual baseline shows that high-frequency or multi-scale information is the remaining bottleneck.

## 🔗 Why EEG and fNIRS tokens can align

### Complementary states, not identical states

EEG and fNIRS observe different coordinates and timescales. EEG tokens should primarily cover neural/electrical states; fNIRS tokens should cover hemodynamic states. Their relationship is:

\[
p(K_t^F\mid K_{t-L:t}^E)
=
\int p(K_t^F\mid H_t)
p(H_t\mid R_{t-L:t},S_{t-L:t})
p(R_{t-L:t},S_{t-L:t}\mid K_{t-L:t}^E)
\,dR\,dS\,dH
\]

This integral explains why the expected mapping is one EEG sequence to a distribution over future fNIRS tokens.

### Physiological signatures

Each EEG codeword receives a teacher signature:

\[
\mu_i^E=E[(R,S)\mid K^E=i]
\]

Each fNIRS codeword receives:

\[
\mu_j^F=E[(\delta f,\delta HbO,\delta Hb)\mid K^F=j]
\]

The physical teacher defines the dynamics connecting these two signature spaces. Stable correspondence means that a sequence of EEG signatures yields a reproducible conditional distribution over fNIRS signatures after controlling history and marginals. It does not mean `i = j`.

### Incremental evidence

Let `H_t^F` denote available fNIRS history and nuisance controls. The physiological-coupling statistic is:

\[
\Delta\ell_t=
\log p(K_t^F\mid K_{t-L:t}^E,H_t^F)
-\log p(K_t^F\mid H_t^F)
\]

A positive global mean is insufficient. Evidence must be positive on held-out subjects and remain positive within prespecified dataset/task scopes. Time-shift and spatial-null controls must remove the gain.

### Preserve, discover, certify

Discrete tokenization is lossy. If tokenizer objectives never test delayed
cross-modal predictability, foundation pretraining cannot recover coupling
information that the codebooks have already discarded. Conversely, forcing a
complete EEG-sequence-to-fNIRS mapping into fixed codeword identities would
collapse context-dependent dynamics into the tokenizer and risk encoding the
teacher's assumptions as the result.

The scientific responsibility is therefore divided as follows:

| Stage | Responsibility | Not sufficient for |
| --- | --- | --- |
| Tokenizer | Preserve local physiological signatures, private information, and broad delayed EEG-to-fNIRS predictive information | Discovering the final contextual law or proving coupling |
| Foundation model | Learn multi-horizon, context-dependent sequence organization with explicit fNIRS-history baseline and EEG-incremental objectives | Serving as its own unbiased certificate |
| Frozen evaluator | Estimate held-out `q_1-q_0` gain with nuisance and null controls | Changing token identity or rescuing a lossy tokenizer |

The training-only preservation shaper is intentionally asymmetric. Its
coupling gradient reaches the EEG semantic tokenizer, while the future fNIRS
target, fNIRS tokenizer, fNIRS-history baseline, and joint teacher are
detached. This tests information preservation without allowing both sides to
collude. A new frozen or cross-fitted evaluator is required after model
selection.

## ⚙️ How an optional target can change semantics

### Waveform target versus state target

The current cache pathway supervises decoded waveforms:

\[
\hat X_{src}^m\approx X_{src,PF}^m
\]

This strongly constrains the decoder output but leaves many latent/codebook organizations equivalent. A flexible decoder can reconstruct the PF waveform even when codeword identity has no stable physical meaning.

A validated auxiliary target may add:

\[
G_m(e_{K_t^m})\approx \mu_t^m
\]

which constrains each prototype to cover a target-defined region. This can be stronger at the semantic bottleneck, but the interpretation is scoped to target family `j` and is not an architecture-level truth.

| Property | Reconstruction/self-supervision | Optional target supervision | Target decision |
| --- | --- | --- | --- |
| Supervised object | Measured EEG/fNIRS or masked/context objective | Family-specific signature and optional uncertainty | Use only targets admitted for the named experiment |
| Where constraint acts | Mainly decoder output | Continuous semantic latent and codebook prototype | Constrain the bottleneck explicitly |
| Constraint dimensionality | High-dimensional and pointwise | Low-dimensional and structured | State target is weaker on samples, stronger on meaning |
| Equivalent latent solutions | Many rotations/code permutations can reconstruct equally | Fewer solutions if prototypes must decode the same state coordinates | Measure prototype/state stability across seeds |
| Treatment of inverse uncertainty | Hidden in one cached estimate | Uniform standardized loss by default; calibrated covariance is an explicit later mode | Do not convert uncalibrated confidence into gradient scale |
| Misspecification risk | Forces the teacher waveform decomposition into the decoder | Can over-organize tokens around an incorrect physical model | Preserve residual and compare shuffled/self-supervised controls |
| Physiological claim supported alone | Clean-component reconstruction only | Teacher-defined state-region discretization after validation | Neither alone proves causal coupling |

An auxiliary target is therefore not “stronger” in every sense. It is more prescriptive about selected semantic coordinates and can be actively harmful when misspecified. The teacher-free measurement objective is the mainline; hybrid objectives are named ablations until their scoped gates pass.

### Uncertainty weighting

When calibrated, target uncertainty defines which examples should strongly organize the codebook:

\[
\mathcal L_{state}^m=
(\hat u_t^m-\mu_t^m)^\top
(\Sigma_t^m+\epsilon I)^{-1}
(\hat u_t^m-\mu_t^m)
\]

Low-confidence targets receive weaker influence. Uncalibrated uncertainty cannot be used merely because a covariance field exists.

For the accepted adaptive physical teacher, the default is
coordinate-standardized, uniform weighting. The equation above becomes active
only after coverage and ranking calibration pass for the exact coordinate and
entrance. A posterior variance field is otherwise diagnostic metadata, not a
training weight.

### Privileged-information boundary

An optional joint target generator may use EEG and fNIRS during training. The modality student must use only its own input. This is privileged-information distillation, not cross-modal inference leakage, provided that:

1. teacher outputs are stop-gradient;
2. EEG and fNIRS students have independent forward paths;
3. coupling evaluation uses independently produced tokens;
4. teacher targets and hyperparameters are fitted without test-subject information.

The full joint teacher posterior need not be recoverable from either modality
alone. Requiring that property would replace the intended multimodal-consensus
estimand with a single-modality translation estimand. The distinction is:

| Question | Required evidence |
| --- | --- |
| May the accepted joint physical teacher supervise a tokenizer experiment? | Complete E0 pass, stable train-only target generation, declared gauge/support, target observability, and non-degenerate finite-vocabulary geometry |
| Are all physical-teacher parameters uniquely recovered? | Parameter/state identifiability and competing-model controls; E0 acceptance alone is insufficient |
| Does EEG contain incremental information about future fNIRS tokens? | Frozen independently generated tokens, fNIRS-history/marginal controls, subject holdout, and time/spatial nulls |

Consequently, poor EEG-only reconstruction of the teacher's complete fNIRS
trajectory is a translation/identifiability diagnostic, not an automatic veto
of a joint privileged teacher. It remains decisive only for claims that require
EEG-only recovery.

## 🔍 Identifiability and competing explanations

### Shared/private non-identifiability

The decomposition:

\[
X^m=X_{semantic}^m+X_{residual}^m
\]

is not uniquely identified by reconstruction. Information can move between branches while preserving the sum. State supervision, bottleneck capacity, uncertainty, branch ablations, and nuisance probes reduce but do not eliminate this ambiguity.

### Teacher misspecification

The physical teacher can be wrong because of fixed parameters, local lead fields, optical Jacobians, noise assumptions, or an insufficient state dimension. The residual branch is therefore a scientific safeguard: it prevents teacher misspecification from forcing information deletion.

### Dataset and source confounding

If dataset source identifies task family, strong source prediction can masquerade as task representation. All task claims require within-dataset or otherwise nuisance-controlled tests. Combined-dataset global accuracy is secondary evidence.

### Window-history mismatch

The fNIRS response at the start of a crop can depend on EEG before the crop. Coupling losses must either supply sufficient history, use full-session context, or mask targets without visible causal support.

## 📊 Falsifiable claims

| Claim | Required observation | Observation that falsifies it |
| --- | --- | --- |
| Semantic tokens retain a registered signature | Prototype-to-signature error beats teacher-free and shuffled-target controls | No improvement or unstable signatures across seeds |
| Residual preserves omitted information | Semantic plus residual recovers task/reconstruction information lost by hard ID | Residual adds no information or only source leakage |
| EEG sequence predicts fNIRS response | Held-out incremental NLL gain over the matched fNIRS-history/marginal and nuisance-controlled baseline | Gain disappears under subject holdout or after source/history/marginal controls |
| Correspondence is physiological | Gain peaks at plausible lags and is destroyed by time/spatial nulls | Gain survives nulls or follows dataset position only |
| Tokens generalize | Registered signatures and task utility remain stable across subjects and seeds | Token matching is arbitrary and downstream gains are source-specific |
| Paired optical input is informative | It improves teacher state confidence or downstream retention over highWL-only | No reproducible improvement under matched capacity |

## 🔐 Allowed and prohibited paper language

### Allowed after the corresponding gates pass

- “We use a sign-calibrated, physiology-constrained adaptive SSM physical
  teacher for independent EEG and fNIRS tokenizers.”
- “The tokenizer discretizes teacher-defined neural and hemodynamic state regions.”
- “EEG token sequences provide incremental held-out information about future fNIRS token distributions.”
- “In a secondary analysis, coupling patterns differed across the examined task conditions.” This language requires direct uncertainty and confound-control evidence but is not a primary gate claim.
- “Soft token posteriors and residual representations preserve information not available from hard IDs.”

### Prohibited without additional causal evidence

- “A specific EEG token causes a specific fNIRS token.”
- “Equal token indices represent the same physiological state.”
- “The residual branch contains only noise.”
- “A non-uniform coupling heatmap proves neurovascular coupling.”
- “Global mixed-dataset coupling is task-invariant.”

## 🔗 References

[^1]: Croce, P., Zappasodi, F., Merla, A., & Chiarelli, A. M. (2017). “Exploiting neurovascular coupling: a Bayesian sequential Monte Carlo approach applied to simulated EEG fNIRS data.” *Journal of Neural Engineering*. https://pubmed.ncbi.nlm.nih.gov/28504643/

[^2]: Foumani, N. M., et al. (2024). “EEG2Rep: Enhancing Self-supervised EEG Representation Through Informative Masked Inputs.” https://arxiv.org/abs/2402.17772

[^3]: Baevski, A., et al. (2022). “data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language.” *Proceedings of Machine Learning Research*. https://proceedings.mlr.press/v162/baevski22a.html

[^4]: Jiang, W.-B., Zhao, L.-M., & Lu, B.-L. (2024). “Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI.” https://arxiv.org/abs/2405.18765

[^5]: Barmpas, K., et al. (2025). “NeuroRVQ: Multi-Scale EEG Tokenization for Generative Large Brainwave Models.” https://arxiv.org/abs/2510.13068

_Last updated: 2026-07-19_
