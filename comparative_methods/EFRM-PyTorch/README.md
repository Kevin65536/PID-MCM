# EFRM synchronous-data reproduction

This directory is the isolated project adaptation of EFRM for the four
synchronously acquired EEG-fNIRS datasets in this repository. The untouched
upstream checkout remains in
`../EFRM-A-Multimodal-EEG-fNIRS-Representation-learning-Model/` at revision
`a62bf3d4c092ac3022b6c0bad90ec3993d5a5720`.

The reporting name is **`efrm_sync_200_10_variable_channel_v1`**. It is a
source-faithful retraining of the EFRM objectives, not a claim of numerical
reproduction: the original checkpoint and processed pretraining arrays were
not released, all project data are paired, the sampling rates are changed to
the repository-wide 200/10 Hz contract, and channels are not duplicated.

## Frozen scientific and data contract

### What the source pretraining corpus actually aligns

The paper's nominal 1,247.5 hours are not 1,247.5 hours of synchronized
multimodal data. Its reported composition is 868 hours / 766 participants of
EEG-only data, 364 hours / 123 participants of fNIRS-only data, and only 15.5
hours / 29 participants of paired EEG-fNIRS data. The released pretraining
loop draws all three loader types each step: the EEG-only branch supplies EEG
masked reconstruction, the fNIRS-only branch supplies fNIRS masked
reconstruction, and only the paired branch supplies the symmetric CLIP loss.
Shorter loaders are cycled, so this is objective-level joint training, not a
claim that unpaired recordings were aligned across people or experiments.

Our corpus has no comparable single-modality expansion. Every admitted sample
is a genuinely synchronized pair, and that one pair supplies both masked
reconstruction losses and the CLIP loss. This preserves EFRM's three
objectives while changing the data regime from `mixed_unpaired_plus_paired`
to the explicitly named `synchronized_in_domain_only` track.

### fNIRS entrance

EFRM consumes the `homer2_aligned_fnirs` branch exposed by
`UnifiedPhysiologyWindowDataset`. Every admitted sample has explicit HbO/HbR
component roles and is transformed to a dimensionless, full-record robust
coordinate before cropping. This is an acceptable EFRM input because the
model reconstructs and aligns a consistent two-component coordinate.

The claim boundary is important. The original physical measurements are not
all connected by a purely linear transform: Single-Trial intensity is mapped
through `-log` optical density and MBLL, whereas REFED, Visual, and
Simultaneous enter from released chromophore exports. Record-level centering
and scaling after component construction are affine and preserve each native
measurement provenance; they do not establish identical physical units.

### Sampling and patch geometry

- EEG: 200 Hz, 8-second pretraining window, 50-sample patch (0.25 seconds).
- fNIRS: 10 Hz, 8-second pretraining window, 20-sample patch (2 seconds).
- The temporal token grids remain 32 EEG patches and 4 fNIRS patches, exactly
  matching the physical patch durations and token counts of EFRM at 128/16 Hz.
- All pretraining examples are real synchronized pairs. The same admitted pair
  supplies EEG MAE, fNIRS MAE, and symmetric EFRM/CLIP alignment losses.
- Batches are grouped by the real measured channel inventory and draw windows
  round-robin across records. Dataset sampling is balanced before selecting
  deterministic epoch-dependent crops. This keeps tensors stackable without
  copying channels while preventing every CLIP negative from being an adjacent
  or overlapping window in the same record. Within-record negatives are still
  measured and reported when they occur.

### Variable-channel policy

All measured channel slots are retained. HbO and HbR are paired by spatial
channel name, and a location is valid only when both components are present
and pass QC. Bad channels and invalid time support are zeroed and represented by masks; they
are excluded from reconstruction loss, self-attention, and global pooling. Batches contain one
record inventory, while sinusoidal position embeddings are generated for the
current channel/time grid. No channel is copied, mirrored, or assigned
fabricated geometry.

This is the primary shared-benchmark variant. An upstream-style channel
duplication experiment may be added only as a separately named sensitivity
analysis and may never replace it.

## Model and loss contract

The default architecture retains the two independent EFRM ViT-base MAEs:
embedding 768, 12 encoder blocks, 12 heads, 512-dimensional decoder, 8 decoder
blocks, 16 decoder heads, and 50% masking. It contains approximately 223M
parameters before downstream heads.

For a synchronized batch `(eeg, fnirs)`:

```text
L = L_EEG_masked_reconstruction
  + L_fNIRS_masked_reconstruction
  + L_symmetric_pair_retrieval
```

The contrastive term preserves the released implementation's fixed `0.1`
similarity multiplier as the primary source-code setting. Raw cosine
similarities are exported separately so this unusual scale cannot obscure the
diagnostic. A conventional learned/divisive temperature is a named ablation,
not the primary EFRM run.

## Leakage boundary and evaluation matrix

Development uses the public subject-grouped train/validation manifests already
frozen for STA-Net. A single development checkpoint is pretrained on public
training subjects from all four datasets; validation subjects are used only
for self-supervised stopping and downstream model selection. Reserved test
subjects are never loaded.

Formal cross-subject evaluation retrains one EFRM checkpoint for every outer
fold and excludes that fold's protected subjects from pretraining, tuning,
normalization, and stopping. Formal within-subject evaluation likewise
excludes the protected session/record/video/trial group from self-supervised
pretraining. A global all-subject checkpoint, if generated for exploratory
embedding plots, is labeled `transductive_diagnostic` and cannot enter the
primary result table.

The downstream matrix matches STA-Net:

| Task | Target | EFRM input support | Reporting name |
| --- | --- | --- | --- |
| Motor imagery | LMI/RMI | paired 8 s | `efrm_sync_classification` |
| Mental arithmetic | MA/BL | paired 8 s | `efrm_sync_classification` |
| Word generation | WG/BL | paired 8 s | `efrm_sync_classification` |
| N-back | 0/2/3-back | paired 8 s | `efrm_sync_classification` |
| DSR | Go/No-go | 2 s EEG plus synchronized 2 s fNIRS context | `efrm_sync_dsr_context_adapter` |
| Visual motivation | RR/RF/FF/FR | paired 8 s | `efrm_sync_classification` |
| REFED | 20-point valence/arousal sequence | paired 20 s | `efrm_sync_regression_adapter` |

Every task reports both frozen-backbone linear probing and full fine-tuning.
Classification selects checkpoints by validation macro-F1 and reports macro
F1, accuracy, balanced accuracy, Cohen's kappa, per-class recall, calibration,
and subject-level uncertainty. REFED selects by masked scaled RMSE and reports
native-coordinate CCC, MAE, RMSE, R2, Pearson/Spearman correlation, coverage,
and subject-level uncertainty. Protected results are opened only through the
shared protocol's explicit unlock path.

Development tuning follows the active STA-Net policy: seed 42, 12 Optuna trials
per task and transfer mode, 2/8/20/40/100-epoch rungs, and best validation
checkpoint through each rung. Formal evidence uses seeds 17, 42, and 73 after
the configuration is frozen.

## CLIP alignment evidence and physiological-coupling contrast

Every validation evaluation exports sample IDs, dataset/subject/record/task,
window times, both 768-dimensional embeddings, raw cosine similarity, scaled
logits, and the exact positive-pair mask. The visualization suite generates:

1. an EEG-by-fNIRS similarity matrix with diagonal positives explicitly boxed;
2. positive, all-negative, within-record, and hard-negative distributions;
3. EEG-to-fNIRS and fNIRS-to-EEG positive rank, top-k retrieval, and MRR;
4. a paired embedding projection with a connector for every true pair;
5. a side-by-side contrast panel: EFRM's symmetric window-retrieval matrix
   versus the project model's directional, lag-conditioned EEG-to-fNIRS
   coupling evidence.

The figure caption must state that an EFRM diagonal is defined by acquisition
co-occurrence and does not identify direction, hemodynamic delay, or a
physiological mechanism. Figure source arrays, pair metadata, SVG, 300-DPI PNG,
and metric JSON are retained together.

## Execution stages and gates

1. **Protocol/preflight:** hash upstream, clean cache, event index, geometry,
   task contracts, public splits, channel inventories, eligible synchronized
   duration, and all exclusions.
2. **Correctness smoke:** tiny model forward/backward on real paired samples;
   prove patch counts, positive diagonal, reconstruction masks, deterministic
   crops, and finite gradients.
3. **Architecture smoke:** full ViT-base on one 4090 with bf16, activation
   checkpointing, and automatic microbatch search. Preserve a true 32-pair
   contrastive matrix with two-pass gradient caching when the model must be
   recomputed in smaller chunks. Ordinary gradient accumulation does **not**
   enlarge CLIP's negative set and is therefore not presented as equivalent.
4. **Development pretraining:** up to 100 epochs, AdamW `1e-4`, betas
   `(0.9,0.95)`, weight decay `0.01`, cosine decay, gradient norm 5, early stop
   patience 15 after at least 20 epochs. Save latest/best and resumable sampler
   state after every epoch.
5. **Transfer and visualization:** run seven-task linear-probe/full-finetune
   public development studies and generate validation-only reports.
6. **Formal evaluation:** freeze configs and rerun pretraining per protected
   outer fold before the explicit shared-protocol test unlock.

The two RTX 4090 GPUs are currently occupied by STA-Net HPO. EFRM GPU work must
wait for that study to complete; CPU correctness tests and artifact construction
may proceed meanwhile.

Current implementation status (2026-07-22): the independent model, unified
adapter, seven-task heads, public-split boundary, CLIP evidence exporter, and
real-data CPU smoke are implemented. Ten unit tests pass; the Single-Trial
smoke completed a finite forward/backward pass with 30 measured EEG channels
and 36 measured HbO/HbR locations. Formal ViT-base architecture smoke and all
performance training remain unopened.

### Parameter and memory audit

The synchronized implementation has 221,459,034 trainable parameters:
110,735,922 in the EEG MAE and 110,723,112 in the fNIRS MAE. The official
default source model has 222,780,000 total parameters, of which 221,466,720
are trainable; most of the total-count difference is its fixed positional
tables. FP32 parameters occupy 0.825 GiB. FP32 parameters, gradients, and the
two AdamW moment tensors require approximately 3.30 GiB before activations;
a weights-only checkpoint is approximately 0.825 GiB and a checkpoint with
Adam state approximately 2.48 GiB before serialization overhead.

Measured on an RTX 4090 with a true 32-pair contrastive matrix:

| Recompute chunk | Activation checkpointing | Peak allocated | Peak reserved | Train+validation smoke |
| ---: | --- | ---: | ---: | ---: |
| 8 | on | 4.20 GiB | 4.55 GiB | 35.8 s |
| 32 | on | 4.19 GiB | 4.38 GiB | 29.1 s |
| 32 | off | 20.25 GiB | 20.62 GiB | 20.8 s |
| 16 | off | 11.89 GiB | 12.55 GiB | 21.9 s |

The development default is therefore batch 32, exact two-pass gradient cache,
recompute chunk 16, BF16 autocast, and no activation checkpointing. This leaves
substantial margin beside the active STA-Net jobs without paying the slower
checkpoint-recompute path. Batched attention masks exclude invalid channel/time
tokens as keys and values without falling back to per-sample transformer loops.

## Artifact contract

Runs live only below `runs/<protocol_id>/...` and follow the common comparison
layout: protocol/method/adapter/split manifests, resolved config, environment,
checkpoints, metrics, predictions, `figures/`, `figure_data/`, hashes, status,
and a self-contained summary. Vendor source files are never modified.

The pretraining entry point is `train_pretrain.py`. It writes resumable
latest/best checkpoints, epoch/step JSONL, the resolved config, split-boundary
hashes, CUDA peak memory, and validation-only CLIP evidence. Its two-pass
gradient cache is regression-tested against the full contrastive-batch
gradient before GPU training.
