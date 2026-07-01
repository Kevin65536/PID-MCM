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

The target architecture does not require this five-state model to be exact. It requires the teacher to provide a testable, uncertainty-aware low-dimensional account of the shared neural-to-hemodynamic dynamics.[^1]

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
| Hard ID | State-region membership | Counting, transition statistics, interpretability |
| Posterior | Boundary uncertainty and secondary states | Robust coupling and downstream inference |
| Prototype embedding | Geometry among state regions | Transfer to sequence models |
| Residual latent | Model-private physiology and reconstruction detail | Information preservation |
| Contextual sequence state | Duration, transition, and spatial grammar | Fine-grained task prediction |

### Sufficient-state condition

Assume the task decision depends on a physiological trajectory `S_1:T` through a score function `g`. If `g` is `L`-Lipschitz and the correct class has margin `gamma`, then the decision is unchanged whenever:

\[
\|\hat S_{1:T}-S_{1:T}\| < \frac{\gamma}{2L}
\]

This is not a guarantee that the tokenizer will meet the bound. It identifies the correct empirical question: compare teacher-state and task-decision error against quantization distortion rather than relying on signal reconstruction alone.

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

## ⚙️ Why the physical teacher changes semantics

### Waveform target versus state target

The current cache pathway supervises decoded waveforms:

\[
\hat X_{src}^m\approx X_{src,PF}^m
\]

This strongly constrains the decoder output but leaves many latent/codebook organizations equivalent. A flexible decoder can reconstruct the PF waveform even when codeword identity has no stable physical meaning.

The state teacher adds:

\[
G_m(e_{K_t^m})\approx \mu_t^m
\]

which constrains each prototype to cover a teacher-state region. The state target is lower dimensional and less restrictive at the waveform surface, but stronger at the semantic bottleneck.

| Property | Cached source-waveform supervision | Physical-state teacher supervision | Target decision |
| --- | --- | --- | --- |
| Supervised object | Decoded clean EEG/fNIRS waveform | Posterior state summary and uncertainty | Use state for semantics; waveform for fidelity |
| Where constraint acts | Mainly decoder output | Continuous semantic latent and codebook prototype | Constrain the bottleneck explicitly |
| Constraint dimensionality | High-dimensional and pointwise | Low-dimensional and structured | State target is weaker on samples, stronger on meaning |
| Equivalent latent solutions | Many rotations/code permutations can reconstruct equally | Fewer solutions if prototypes must decode the same state coordinates | Measure prototype/state stability across seeds |
| Treatment of inverse uncertainty | Hidden in one cached estimate | Explicit covariance/validity weighting | Down-weight ambiguous states |
| Misspecification risk | Forces the teacher waveform decomposition into the decoder | Can over-organize tokens around an incorrect physical model | Preserve residual and compare shuffled/self-supervised controls |
| Physiological claim supported alone | Clean-component reconstruction only | Teacher-defined state-region discretization after validation | Neither alone proves causal coupling |

The state teacher is therefore not “stronger” in every sense. It is less prescriptive about exact waveform samples but more prescriptive about the semantic coordinates represented by the codebook. The hybrid objective intentionally uses these non-equivalent constraints together.

### Uncertainty weighting

Teacher posterior uncertainty defines which examples should strongly organize the codebook:

\[
\mathcal L_{state}^m=
(\hat u_t^m-\mu_t^m)^\top
(\Sigma_t^m+\epsilon I)^{-1}
(\hat u_t^m-\mu_t^m)
\]

Low-confidence teacher states receive weaker influence. This prevents ambiguous inverse solutions from acting as exact labels.

### Privileged-information boundary

The joint teacher may use EEG and paired optical observations during training. The modality student must use only its own input. This is privileged-information distillation, not cross-modal inference leakage, provided that:

1. teacher outputs are stop-gradient;
2. EEG and fNIRS students have independent forward paths;
3. coupling evaluation uses independently produced tokens;
4. teacher targets and hyperparameters are fitted without test-subject information.

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
| Semantic tokens retain physical state | Prototype-to-state error beats reconstruction-only and shuffled-teacher controls | No improvement or unstable signatures across seeds |
| Residual preserves omitted information | Semantic plus residual recovers task/reconstruction information lost by hard ID | Residual adds no information or only source leakage |
| EEG sequence predicts fNIRS response | Held-out incremental NLL gain over fNIRS history/marginal baseline | Gain disappears within dataset/task or under subject holdout |
| Correspondence is physiological | Gain peaks at plausible lags and is destroyed by time/spatial nulls | Gain survives nulls or follows dataset position only |
| Tokens generalize | Physical signatures and task utility remain stable across subjects and seeds | Token matching is arbitrary and downstream gains are source-specific |
| Paired optical input is informative | It improves teacher state confidence or downstream retention over highWL-only | No reproducible improvement under matched capacity |

## 🔐 Allowed and prohibited paper language

### Allowed after the corresponding gates pass

- “The tokenizer discretizes teacher-defined neural and hemodynamic state regions.”
- “EEG token sequences provide incremental held-out information about future fNIRS token distributions.”
- “Coupling patterns differ across prespecified task conditions.”
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

_Last updated: 2026-07-01_
