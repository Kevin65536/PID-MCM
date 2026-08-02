# NormWear identity and adaptation audit

This audit freezes the source, executable checkpoint, input-rate decision, and
representation boundary before target performance is inspected. All results
must be reported as `normwear_eeg_fnirs_adapted`.

## Identity chain

| Source | Frozen identity |
| --- | --- |
| Official GitHub checkout | `07517fcb13def8c89cb586128359cec02f86ec8d` |
| Official v1.0.0-alpha backbone | SHA-256 `36d0bca18356ccfc8e8916058bf838f26f1212a646f5780b487ad78581a92561`, 544,579,503 bytes |
| Root license | Apache-2.0; SHA-256 `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4` |

The checkpoint is a direct 261-entry `OrderedDict`, not a training wrapper. A
safe `weights_only=True` load contains 136,116,425 tensor elements and strictly
matches the pinned `NormWear` construction. It includes the full pretraining
decoder, but the adapted comparison loads and executes only the encoder. The
optional MSiTF text-alignment checkpoint is outside the primary track.

The repository root grants Apache-2.0. `modules/normwear.py` also carries an
“All rights reserved” copyright header and explicitly points to the root
license; this notice is retained as source provenance rather than silently
rewritten.

## Input-rate and measurement decision

The shared canonical arrays are EEG at 200 Hz and paired HbO/HbR at 10 Hz. The
official README example says 64 Hz, while `NormWearModel.sampling_rate` and the
pretraining image width are fixed at 65. The official helper only resamples
rates above 256 Hz, so it would silently leave both project rates unaligned.

This adapter therefore uses deterministic polyphase anti-alias resampling to
the executable model's 65 Hz coordinate: EEG uses 13/40 and fNIRS uses 13/2.
No labels or fitted statistics enter this operation. The upstream
`basic_preproc` path is not stacked on the canonical record-wise robust
coordinate because it is not used by the official embedding example and would
introduce another amplitude normalization.

Every supported sample preserves all real measured channels. Delivered order
is frozen as EEG, then HbO, then HbR, with modality-prefixed channel identities.
No missing channel or time sample may be copied, padded, mirrored, or generated.

## Representation decision

The adapter preserves the optimized upstream Ricker CWT, its aligned signal /
first-difference / second-difference planes, scales 0.1 through 64, the 9 by 5
non-overlapping patch projection, the interpolated pretrained positional grid,
all 12 encoder layers, and the cross-channel CLS fusion at even layers.

The final layer-normalized token tensor has shape
`[batch, real_channels, patches_plus_cls, 768]`. Following the official README
example, tokens are mean-pooled within each channel and the channel vectors are
concatenated in the frozen delivered order. The feature width is therefore
`real_channels * 768`; there is no adaptation-specific cross-modality average.
Only the outer-training linear probe is trainable.

The implementation chunks the channel-independent attention calls while
retaining complete cross-channel CLS fusion. The unchunked path is bitwise
identical to the pinned upstream call. The retained GPU smoke requires token
maximum/mean absolute differences below 0.01/0.0002, pooled-feature maximum
absolute difference below 0.0002, and pooled-feature cosine above 0.99999 for
the chunked float32 path.

## Cell registration

Motor imagery, mental arithmetic, WG, n-back, DSR, and visual classification
are registered as support-matched direct cells before performance inspection.
DSR consumes exactly two synchronized seconds; its fNIRS input is described as
block-anchored context, never as an event-native hemodynamic response.

REFED regression is preregistered unsupported. Its partial terminal support is
scientifically real, but the released encoder has no temporal support-mask
path and the released downstream code has no masked sequence-regression
contract. Padding or method-specific deletion would violate the shared
estimand.
