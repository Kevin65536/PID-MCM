# Physiology-semantic tokenizer code migration plan

_Decision-complete migration boundary for P2 through P5; P6 coupling is excluded_

> Maintenance note: subsequent modification plans must annotate their affected components on a plan-specific copy of the maintained SVG architecture. See [`08_ARCHITECTURE_VISUALIZATION.md`](08_ARCHITECTURE_VISUALIZATION.md).

---

## 📋 Objective and execution boundary

This migration replaces the archived reconstruction-centered `source_observation` mainline with independently inferable EEG and fNIRS semantic tokenizers. The migration is complete at P5 export compatibility. It does not implement or evaluate P6 sequence coupling.

The following rules are fixed for all new work after the 2026-07-14 input-contract revision:

- every new run gets measured EEG/fNIRS from `UnifiedPhysiologyWindowDataset`;
- formal validation of an optional target starts only after its adapter passes deterministic correctness tests and a real-data dry run;
- target-supervised training does not start before its scoped E0 establishes valid prediction and admissible coordinates;
- an explicit teacher-free reconstruction-plus-VQ baseline may optimize after a blocked E0 because it consumes no teacher target;
- legacy aliases, four-branch source/observation quantization, and pre-VQ cross-modal exchange remain archive-only;
- smoke success is software evidence, not a scientific gate decision.

```mermaid
flowchart LR
    accTitle: P2 Through P5 Migration Order
    accDescr: Code migration begins with isolated quantizer and teacher contracts, permits E0 only after teacher correctness, and then proceeds through tokenizer training and export integration.

    p1(["P1 data contract"]) --> p2["P2 corrected quantizer"]
    p1 --> p3["P3 teacher adapter"]
    p3 --> check{"P3 correctness passed?"}
    check -->|No| repair["Repair contract"]
    repair --> p3
    check -->|Yes| e0["E0 teacher validity"]
    p2 --> p4["P4 tokenizer and losses"]
    e0 --> p4
    p4 --> p5(["P5 export and consumers"])

    classDef foundation fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#1e3a5f
    classDef decision fill:#fef9c3,stroke:#ca8a04,stroke-width:2px,color:#713f12
    classDef valid fill:#dcfce7,stroke:#16a34a,stroke-width:2px,color:#14532d
    classDef blocked fill:#fee2e2,stroke:#dc2626,stroke-width:2px,color:#7f1d1d

    class p1,p2,p3,p4 foundation
    class check decision
    class e0,p5 valid
    class repair blocked
```

## 📥 Runtime data contract

`UnifiedPhysiologyWindowDataset` with contract `unified_physiology_window_v1` is the only active measured-data entry for newly planned experiments. The earlier `physiology_semantic_v2` Croce cache contract remains a historical/optional target adapter.

| Field | Shape | Consumer | Inference input |
| --- | ---: | --- | :---: |
| `eeg` | `[B, C_E, 4000]` | EEG tokenizer | Yes |
| `fnirs` | `[B, C_F, 200]` | fNIRS tokenizer | Yes |
| `valid_mask.*` | modality time axis | Tokenizer, losses and export | No |
| `alignment` | Sample metadata | Temporal model and audit | No |
| `label` | Canonical mapping | Probe/downstream suites | No |
| `channel_geometry` | Variable row lists | Spatial adapter and audit | No |
| `auxiliary_target.*` | Family-specific and optional | Named target adapter/loss only | No |
| normalization/provenance | Sample metadata | Audit and export | No |

The default 20-second context may be divided into ten non-overlapping two-second patches:

- EEG: `[B, 10, C_E, 400]`;
- fNIRS: `[B, 10, C_F, 20]`.

The model-facing APIs are `encode_eeg(eeg)` and `encode_fnirs(fnirs)`. Passing the other modality, auxiliary targets, or decomposition fields is a schema error.

## 🧠 Temporal representation boundary

Token identity is patch-local. The pre-quantization encoder may transform samples and channels inside one two-second patch, but it must not attend to another patch and must not use an absolute crop position. This makes a token invariant to where its source patch appears inside a crop.

Sequence learning is a separate post-quantization responsibility. For a target at patch `t`, the context module consumes only patches `[t-5, ..., t-1]`, representing ten seconds of prior history. Targets `t < 5` have incomplete history and contribute zero masked-context loss. The context output never replaces or mutates the exported token identity.

This boundary follows the separation between local latent quantization and contextual prediction in wav2vec 2.0,[^1] and the frozen-tokenizer-then-causal-model division used by NeuroLM.[^2] LaBraM remains the source for generic EEG patch and spectral-tokenizer components, but its full pre-quantization sequence Transformer is not copied because it would make token identity depend on neighboring and future patches.[^3]

## ⚙️ P2 corrected quantizer

Create a new active quantizer module rather than changing `NormEMAVectorQuantizer`, which is still imported by compatibility code.

### Public output

`QuantizerOutput` contains:

| Field | Shape | Definition |
| --- | ---: | --- |
| `logits` | `[B,N,K]` | Negative squared Euclidean distance divided by temperature |
| `posterior` | `[B,N,K]` | Softmax-normalized logits |
| `hard_ids` | `[B,N]` | Posterior argmax |
| `quantized` | `[B,N,D]` | Straight-through hard codebook lookup |
| `expected_embedding` | `[B,N,D]` | Posterior-weighted codebook expectation |
| `commitment_loss` | scalar | Encoder-to-selected-code commitment |
| `health` | dictionary | Entropy, active codes, rank, geometry, drift, and revival count |

Mainline assignment is non-normalized Euclidean distance. Cosine assignment is a named ablation. The resolved configuration asserts `K=128` and `D=64` for both modalities before the first forward pass.

### EMA update

The quantizer stores codebook weights, assignment-count EMA, vector-sum EMA, initialization state, update count, and revival count in its state dict. For each distributed step:

1. all-reduce current assignment counts and vector sums;
2. update both EMA buffers with the configured decay;
3. update only codewords with a positive current global assignment count;
4. leave zero-assignment codewords unchanged;
5. apply a configured, explicitly logged dead-code revival policy after the warm-up interval.

Checkpoint reload must reproduce logits, posterior, IDs, embeddings, EMA buffers, and health counters exactly in evaluation mode.

## 🧪 P3 optional target adapters

Every adapter converts one named target family into deterministic, detached patch/context targets with provenance, uncertainty when calibrated, and a validity mask. No adapter changes the measured inputs or tokenizer inference API.

For each state coordinate and patch, compute:

- mean over the patch;
- least-squares slope against time in seconds;
- `log(mean posterior variance + temporal variance of posterior means + eps)`.

The historical Croce adapter may still expose its six-dimensional EEG and nine-dimensional fNIRS summaries for a named ablation. These dimensions are not the target architecture contract; alternative families define their own versioned shapes and identifiability scope.

A target patch is valid only if every underlying cache-valid and causal-valid fNIRS sample is valid and all required means and variances are finite. The first five patches are therefore ineligible under the fixed ten-second history contract. Invalid targets and their uncertainty weights are finite placeholders with a false mask; they must contribute exactly zero loss.

P3 correctness requires family-specific reference trajectories, uncertainty propagation where applicable, mask contraction, stop-gradient behavior, shape assertions, and a one-sample unified-loader join dry run. Only use of that target family remains blocked until its checks pass.

## 🧩 P4 tokenizer, losses, and training

### Independent modality branches

Each branch performs:

1. reshape raw input into two-second patches;
2. compute frequency-aware patch embedding using the existing generic `MultiChannelPatchEmbedding`;
3. apply a patch-local projection/MLP with no cross-patch operation;
4. split into a 64-dimensional semantic latent and a continuous residual latent;
5. quantize the semantic latent;
6. decode registered signatures from the continuous semantic latent and codebook embeddings when the corresponding objective is enabled;
7. reconstruct normalized raw patches from semantic-plus-residual, semantic-only, and residual-only inputs.

| Modality | Encoder output | Semantic latent | Residual latent | Optional target |
| --- | ---: | ---: | ---: | ---: |
| EEG | 256 | 64 | 64 | family-specific |
| fNIRS | 160 | 64 | 32 | family-specific |

The registry name is `physiology_semantic`. It is registered by the active tokenizer package without importing compatibility registration.

### Loss routing

The loss module exposes independently weighted state, prototype, masked-state, reconstruction, VQ, and private terms. State-related terms are uncertainty weighted and masked. The private term exists in configuration and metrics but defaults to zero until its scientific ablation is specified.

The fixed-history context module predicts state at `t` from the five preceding expected embeddings. It uses relative lag embeddings only. Perturbing patch `t` or any future patch must not change its context prediction.

### Training entry

The active training script supports `--dry-run`, `--smoke`, `--train`, and `--resume`. It implements epoch training and validation, AMP, gradient clipping, AdamW, warm-up/cosine scheduling, early stopping, best/last checkpoints, and complete optimizer/scheduler/scaler state restoration. A run saves resolved configuration, environment, split/cache manifest, JSONL metrics, quantizer health, teacher diagnostics, completion status, and hashes required by the run artifact contract.

The 2026-07-03 E0 pilot blocked the historical Croce physical-state objective, so that exact objective cannot take optimizer steps. The 2026-07-16 design decision separately admits the gauge-corrected adaptive joint proxy for development; its new decision artifact, cache sidecar and loss routing must be implemented before optimizer use. The trainer continues to verify the concrete decision artifact, split hash, data contract, cache roots and admitted coordinates. The teacher-free reconstruction-plus-VQ path remains independently runnable.

## 📤 P5 export and consumer contract

The versioned export stores, per modality:

- hard IDs;
- full posterior or configured top-k posterior;
- expected codebook embedding;
- continuous residual;
- optional family-specific target/signature, uncertainty, and validity mask;
- subject, source, task, anchor, crop, cache schema, checkpoint hash, and sample-order hash.

Consumers expose four explicit modes: `hard`, `codebook`, `soft`, and `semantic_residual`. `codebook` indexes the checkpoint codebook directly; creating a fresh embedding table is an error. Round-trip validation reloads the checkpoint and requires identical IDs and posteriors for the exported sample order.

## ✅ Acceptance tests

| Boundary | Blocking assertion |
| --- | --- |
| Patch locality | Identical patch content at different crop positions yields identical logits, IDs, and embeddings |
| Causality | Future or target-patch changes do not alter a fixed-history prediction |
| Modality independence | Changing EEG cannot alter fNIRS outputs and vice versa |
| Gradient isolation | No branch gradient reaches the other modality encoder or codebook |
| Auxiliary-target mask | Invalid targets contribute exactly zero to that supervised loss |
| Quantizer state | Zero-assignment stability, EMA arithmetic, revival logging, and reload equivalence pass |
| Pipeline | Loader → teacher → tokenizer → loss → checkpoint → export passes on CPU |
| Consumer | Export modes use checkpoint-native codebook geometry and preserve sample order |

The execution sequence is unit tests, integration tests, unified-loader dry run, scoped target validation when applicable, then objective-specific smoke and short formal. A failed target validation blocks only objectives that consume that target. P6 remains blocked until tokenizer freeze and the G2/G3 information-retention and registered-semantics gates pass.

## 🔗 References

[^1]: Baevski, A., Zhou, H., Mohamed, A., & Auli, M. (2020). “wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations.” https://arxiv.org/abs/2006.11477
[^2]: Jiang, W.-B., Wang, Y., Lu, B.-L., & Li, D. (2025). “NeuroLM: A Universal Multi-task Foundation Model for Bridging the Gap between Language and EEG Signals.” https://arxiv.org/abs/2409.00101
[^3]: Jiang, W.-B. et al. (2024). “Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI.” https://proceedings.iclr.cc/paper_files/paper/2024/file/47393e8594c82ce8fd83adc672cf9872-Paper-Conference.pdf

_Last updated: 2026-07-16_
