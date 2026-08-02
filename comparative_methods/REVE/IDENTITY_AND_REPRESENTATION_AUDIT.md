# REVE identity and representation audit

This audit freezes the project adapter boundary independently of the existing
method manifest. It covers the exact executable checkpoint snapshot, the
electrode-position bank, the representation layer, and the trainable-parameter
boundary used for support-matched comparisons.

## Identity chain

REVE has three independently versioned sources and all three are required to
identify a result:

| Source | Frozen identity | Local SHA-256 |
| --- | --- | --- |
| Official GitHub source | `06a7059a07c3dabd80aee60c3dbc1eca4bdbe1c7` | n/a (Git tree) |
| `brain-bzh/reve-base` snapshot | `fa9a2163a4b7c0a42c8e28b56077ef9c368944dc` | weights: `8ecc650619598748286c2457f81f5c6bd12e8bb59db44f7b02af1955c44de8fe`; executable model code: `207a9c51218d806ab317eabf4eb6d3234eabe887f48e541ef5580c4b9b45359f` |
| `brain-bzh/reve-positions` snapshot | `befa5b57a455b77cf302daf610c2e9ed8140bace` | weights: `4b793820b9df0998667deb6c8ce2dbb86b38221d5165c1844bc4971941b13f13`; executable position code: `26ad5d55fb667da7c80bcc8e8512eb35bca33054a3b3e2bcafe4775543aa1cd0` |

The adapter verifies both safetensors files and every executable trusted-code
file before local `AutoModel` loading. The base snapshot's executable
`modeling_reve.py` is not byte-identical to the copy bundled in the pinned
GitHub checkout (GitHub-copy SHA-256
`3292273147a5abc528e5478bab4b29e59ef07aba9c9b9eb12169da8f2f364481`).
Consequently, the Git revision cannot stand in for the Hugging Face snapshot
revision; evidence must retain both identities.

The verified position bank contains 543 unique names, a one-to-one index map,
and a finite `(543, 3)` coordinate tensor. Unknown channel names fail closed.

## Representation decision

The public adapter uses the final transformer latent tokens returned after the
snapshot's identity `final_layer`. It then applies the checkpoint's pretrained
`cls_query_token` attention pooling over the flattened real-channel by
real-patch token grid, producing one 512-dimensional embedding per sample.

This is the released checkpoint's `attention_pooling` route and corresponds to
the compact global representation encouraged by REVE's secondary pretraining
loss. Tests reconstruct the attention equation directly and require numerical
agreement with the adapter.

## Frozen-probe capacity boundary

The pinned upstream downstream code labels its first training stage “linear
probing”, but `freeze_model` leaves both the linear head and the 512-parameter
`cls_query_token` trainable. The shared project protocol instead requires a
static frozen representation and the same trainable linear-head boundary for
BIOT, CBraMod, and REVE. Therefore this adapter freezes the pretrained query
token and trains only the downstream linear head.

This is an explicit optimization-capacity deviation from upstream REVE linear
probing. It is not presented as an exact reproduction of the upstream LP
numbers. Training the query would invalidate the static feature-cache boundary
and give REVE extra method-specific trainable capacity.

## Dataset-coordinate and overlap decisions

The project supplies the same canonical recordwise robust-standard-deviation
EEG coordinate to each support-matched method. Upstream REVE downstream loaders
use task-dependent scale factors (including 10, 100, and 1000), rather than one
model-wide deterministic amplitude transform. Those raw-array scale factors
are therefore not stacked on the already standardized project signal.

REVE receives exactly 200 Hz EEG and official-bank coordinates for the frozen
real measured-channel panel. It refuses padded channels, padded time support,
unknown electrode names, and non-finite values. Its official 200-sample patches
advance by 180 samples; an incomplete final patch is discarded exactly as in
the executable snapshot.

The paper's exhaustive pretraining list includes `Shin2017A`. The project's
Single-Trial dataset is the corresponding EEG-fNIRS corpus, so its REVE result
must remain in the `open_world_pretrained_with_target_corpus_overlap` reporting
track. No target-excluded pretraining claim is permitted for that dataset.

## Frozen adapter contract

- input: `(batch, real_channels, samples)` at exactly 200 Hz;
- geometry: exact channel-name lookup in the verified official position bank;
- token tensor: `(batch, real_channels, patch_count, 512)`;
- pooling: frozen pretrained query attention over real tokens;
- output: finite `(batch, 512)` embedding;
- trainable parameters: downstream linear-head weight and bias only.
