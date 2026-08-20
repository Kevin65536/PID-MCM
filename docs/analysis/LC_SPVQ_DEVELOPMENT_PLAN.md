# LC-SPVQ development plan

_Protocol status: new exploratory method generation; implementation in progress. This plan does not alter the completed 2026-08-19 SSM-reliability or continuous shared/private verdicts._

## 1. Scientific estimand and claim boundary

The new shared construct is a lagged probabilistic channel between two modality-specific state spaces:

\[
Z_E(t) \longrightarrow P\{Z_F(t+\tau)\mid Z_E(t)\}.
\]

It is not coordinate equality, same-ID semantics, or decoder interchangeability. EEG and fNIRS retain independent encoders and independent codebooks. A positive result may support the narrow description **coupling-relevant token** only when a frozen representation improves a held-out cross-modal proper score over fNIRS history, time/task controls, and declared pairing/timing nulls. It cannot establish a causal or mechanistic neurovascular claim.

The completed no-VQ experiment remains a negative result under its original estimand: only `2/16` simultaneous lower bounds exceeded zero, every fNIRS-target cell failed, and every matched-swap cell failed. LC-SPVQ tests a new estimand and must not relabel that family as successful.

## 2. Data-access and analysis status

The first generation is development-only and exploratory.

- Fit subjects are 01–18 in each owning dataset.
- Subjects 01–15 are the default parameter-fit subset.
- Subjects 16–18 are the default fit-only checkpoint-selection subset.
- Subjects 19–23 are pure-apply development evaluation subjects.
- Single-Trial subjects 24–29 and Simultaneous subjects VP024–VP026 remain closed. The unified loader may enumerate their cache metadata, but the LC-SPVQ wrapper filters subject identity before any sample `__getitem__`; protected measured arrays are never dereferenced.
- Existing development subjects have already informed prior method decisions. They are not a new independent confirmatory holdout, even when the new runner applies a frozen model to them.
- Artifact annotations remain diagnostic; recorded and analysis-valid support masks determine admitted points.

Task-specific 20 s windows use a common `-5 s` event/block-relative start and ten non-overlapping 2 s patches. Measured access is bound to the repository-local canonical clean cache, the reviewed `single_trial_eeg_artifact_clean_v4` EEG branch, paired timestamps, and the recorded/analysis-valid support intersection; dataset, subject, record, event identity, and both modality-specific window-start clocks are rechecked when each admitted sample is loaded. The downstream task classes are:

| Task | Dataset | Classes | Records |
| --- | --- | --- | --- |
| Motor imagery | Single-Trial | LMI, RMI | sessions 00/02/04 |
| Mental arithmetic | Single-Trial | MA, BL | sessions 01/03/05 |
| Word generation | Simultaneous | WG, BL | `cnt_wg` |
| N-back | Simultaneous | 0/2/3-back sessions | `cnt_nback` |

Task-specific models use all real measured channels in their frozen channel order. No copied or padded channel is introduced. The older exact 6-EEG/2-fNIRS SSM-selected view is used only by the existing-export lag probe because that is the coordinate its checkpoint actually encoded.

## 3. Stage 0: existing-export lag probe

Before new encoder training, evaluate two paths at lags

\[
\{-4,-2,0,2,4,6,8,10\}\;\mathrm{s}.
\]

1. current continuous EEG shared latent → current continuous fNIRS shared latent;
2. EEG native patch features → fNIRS native patch features.

Probe fitting and all feature normalization use fit subjects only. However, the old `best.pt` checkpoints were already selected with subjects 19–23, and fit latents were not retained in `validation_predictions.npz`. Re-encoding fit samples is therefore a new manifest-bound derived export, and evaluation on 19–23 is explicitly **post-selection development**, not a fresh held-out test. The probe must not consume SSM `target`, `eeg_driver`, or `fnirs_driver` fields. Development evaluation compares:

- the matched trial;
- a deterministic non-identity donor from the same subject and condition at the same token position;
- a non-zero within-trial circular shift;
- negative-lag controls.

The legacy exports do not retain event timestamps, so their deranged null verifies same-subject/same-condition nonidentity but cannot verify 20 s window nonoverlap; no nonoverlap claim is made for that diagnostic. The unit of uncertainty is subject. Window and token rows are prediction observations, not biological replicates. Ridge is the primary low-capacity probe; any CCA result is diagnostic. Because the current shared tokens are full-window bidirectional, a positive positional-lag result is evidence of retained cross-modal information but not evidence that their receptive fields implement a physiological delay.

Decision guide:

| Existing latent | Native features | Development interpretation |
| --- | --- | --- |
| positive-lag matched advantage | any | current representation retained information; proceed without attributing causality |
| no advantage | matched advantage | retrain the shared branch; the previous objective discarded available structure |
| no advantage | no advantage | restrict the claim to task-locked association unless a later independent design establishes trial-specific gain |

## 4. Model variants

### B0: continuous architecture baseline

B0 is a new-generation architecture baseline, not a replay of the frozen SSM-target experiment. It uses four full-window bidirectional continuous encoders, modality-native shared feature prediction, private raw reconstruction, and the downstream task head. It has no lag objective and no VQ. Raw reconstruction gradients remain stopped at the shared branch.

### C1: lag-predictive continuous model

C1 changes only the shared construct:

- EEG shared token: current 2 s patch plus at most one preceding patch;
- fNIRS shared token: current patch plus a configurable two preceding patches (maximum 6 s local history);
- private encoders: full-window bidirectional;
- continuous pre-VQ lag matching at `0/2/4/6/8/10 s`;
- no SSM target and no token-ID/co-occurrence objective.

C1 is retained as a diagnostic bridge even when the first execution matrix prioritizes B0/M1/N1.

### M1: LC-SPVQ main model

M1 adds independent EEG and fNIRS EMA codebooks to a stable C1 checkpoint:

- `K=16`, `D=64` in the primary model;
- train-only continuous-latent k-means initialization;
- soft-to-hard quantization-strength and posterior-temperature schedules;
- hard ID, posterior, expected embedding, annealed/hard quantized embedding, and pre-VQ latent exports;
- no shared IDs, codebook parameters, or token meanings across modalities.

### N1: pairing null

N1 is identical to M1 except that the positive cross-modal training endpoint is replaced by a stable same-subject, same-condition, non-identity trial donor. Native feature prediction, private reconstruction, optimizer, schedules, model selection, and downstream evaluation remain unchanged. N1 is a coupling null, not an alternative model selected for performance.

## 5. Training objectives

### Native shared targets

EEG shared tokens predict a compact train-standardized surface derived locally from the existing physiological feature extractor: theta/alpha/beta/low-gamma log power, spectral entropy, line length, and Hjorth mobility, aggregated only across per-sample channel-valid EEG channels.

fNIRS shared tokens predict component-resolved local morphology: HbO/HbR mean, slope, endpoint difference, and AUC, aggregated within component role only across per-sample channel-valid measured locations. A coordinate is unsupported when no valid contributing channel remains. These targets describe token prototypes; they do not replace measured raw signals.

### Private objective

Each private token reconstructs its own train-normalized raw patch. The private encoder and raw decoder receive this gradient; the shared encoder does not.

### Lag-predictive objective

The primary matching loss is computed on continuous projected shared latents before VQ. It uses the fixed non-negative lag bank and a learned train-only lag distribution. The target branch is stop-gradient or momentum-compatible. For each lag, batch negatives are restricted to same-subject, same-condition, nonidentity trials at the same lag endpoint as the positive relation: `E_trial(t) -> F_other_trial(t+lag)`. The registered donor uses that endpoint-aligned relation, and the reverse-direction mask is its transpose. Historical v2 exports retain the earlier same-token-time null; endpoint-aligned training is emitted only under the v3 export contract. Donor eligibility fails closed unless every group admits a complete permutation whose 20 s windows do not overlap on either the EEG or fNIRS event clock within a record. EEG→fNIRS and fNIRS→EEG terms are reported separately and combined equally.

The first sensitivity is limited to

\[
\lambda_{\mathrm{lag}}\in\{0.1,0.5\}.
\]

No depth, head-count, hidden-width, lag-bank, or codebook-size search is conducted in the same round. A selected value is frozen on fit-only selection subjects before development evaluation.

## 6. Quantization and downstream head

For sample \(n\), the soft lagged coupling tensor is

\[
C^{(n)}_{ij\tau}=\sum_t p_E^{(n)}(i\mid t)p_F^{(n)}(j\mid t+\tau).
\]

A rank-8 bilinear head consumes the tensor without flattening it into a large unrestricted MLP. A separate private head consumes masked pooled continuous private latents. Logits are exported as:

- coupling-only;
- shared marginal-only;
- private-only;
- coupling + private.

During head fitting, shared encoders and both codebooks are frozen. The private branch may be fine-tuned only under a declared configuration; frozen-private and fine-tuned-private results remain separately labeled.

Primary downstream reporting uses macro-F1 with the declared class order. Pooled-window and subject-equal summaries are both retained; neither treats seeds or windows as independent subjects. B0, M1, and N1 use the same split, task samples, head capacity, and early-stopping rule.

## 7. Coupling endpoint and null family

The primary cross-modal endpoint compares held-out categorical proper scores:

\[
q_0: Z_F(t+\tau)\sim Z_F(<t+\tau)+\text{time/task controls},
\]

\[
q_1: Z_F(t+\tau)\sim Z_F(<t+\tau)+Z_E(t)+\text{time/task controls}.
\]

The direction is frozen so that lower loss is better:

\[
\Delta_{\mathrm{cpl}}=\operatorname{loss}(q_0)-\operatorname{loss}(q_1).
\]

Log loss is primary and Brier gain is secondary. Models are fitted on fit subjects and applied without refitting to development subjects. **Numerical implementation amendment (2026-08-20):** the fit-target posterior only is mixed with 5% uniform label mass so absent K16 classes have a finite categorical optimum; held-out targets remain unsmoothed, L2 is fixed at 1.0, and both q0/q1 fits must converge within 5,000 iterations at tolerance 1e-6 or publication fails closed. These values are fixed software regularization, not selected on fit-selection or development outcomes. The subject-level matched gain is compared with:

- same-subject/same-condition trial derangement;
- within-trial non-zero circular shift;
- negative lags;
- the N1 training null.

Token coupling is not established by classification gain alone.

## 8. Coupling-map analysis

Held-out soft counts receive fixed Dirichlet smoothing. The display quantity is conditional log-lift relative to the fNIRS marginal, followed by subtraction of the expected deranged-trial log-lift:

\[
R_{ij\tau}=M^{\mathrm{matched}}_{ij\tau}
-\mathbb E_{\mathrm{deranged}}M_{ij\tau}.
\]

Row and column order, color range, smoothing, and peak-lag selection are learned or fixed on fit subjects and then frozen. Development or later protected data never determines display order. Quantitative summaries precede figure interpretation:

1. positive-versus-negative/null lag specificity;
2. pair concentration;
3. seed/fold/subject-bootstrap top-pair stability;
4. native-feature coherence of top EEG/fNIRS prototypes.

Raw co-occurrence count and a visually attractive heatmap are not success criteria.

## 9. Execution order and readiness gates

1. Complete pure-function and synthetic known-lag tests.
2. Run the existing-export lag probe.
3. Run the registered first-round B0/M1/N1 software smoke tests without protected access; retain C1 as a later sensitivity rather than a first-round gate.
4. Run motor imagery and word generation on the fit/selection/development split.
5. Freeze one configuration using fit-only evidence.
6. Apply the same configuration to mental arithmetic and N-back.
7. Only after a separate independent-holdout protocol exists, consider a formal five-fold or protected campaign.

Measured training is not ready until tests establish:

- future-patch perturbations cannot change earlier shared tokens;
- private tokens may retain full-window context;
- invalid patches cannot enter lag pairs or VQ updates;
- raw loss has no shared-encoder gradient;
- EEG and fNIRS codebooks are distinct objects with distinct buffers;
- lag loss recovers a synthetic known-lag many-to-many process and fails on its deranged/independent null;
- coupling proper-score functions recover a synthetic incremental EEG effect beyond fNIRS history;
- complete JSON serialization, checkpoint reload, cache-to-summary execution, manifest hashing, and atomic publication succeed.

## 10. Frozen first-round outputs

The minimum implementation package is:

- existing-export lag probe, exact null registries, subject-level scores, and manifest;
- B0/M1/N1 reviewed configuration;
- reusable local/causal shared encoder, independent K16 VQ surface, lag objective, coupling head, and private head;
- soft coupling/log-lift/proper-score analysis functions;
- synthetic and software tests;
- smoke-run evidence and commands.

A full development result will be added only after those gates pass. Until then, project status should say **implementation in progress / scientific verdict unreviewed**, not that LC-SPVQ is supported.
