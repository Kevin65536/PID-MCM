# Joint protected comparison campaign results

_Result snapshot: 2026-08-14; campaign execution, unblinding, aggregation, and
cell-level numeric acceptance are complete_

## Campaign terminal status

Campaign `joint-comparison-protected-20260813-v3-single-gpu` completed all
540/540 preregistered jobs with zero failed, invalid-output, missing, or
technical-failure jobs. The execution controller sealed the complete job set at
2026-08-13 21:04:22 CST; the two accountable roles authorized unblinding at
2026-08-14 09:27:44 CST, and the aggregate was created at 2026-08-14 09:29:44
CST. [E3, E4, E5]

Completion does not imply that every numeric result is admissible. Across the
42 registered method-task cells, the frozen acceptance rules assigned 22
`TABLE_READY_WITH_NOTE`, 12 `REJECTED_VALUE`, 2 `OVERLAP_TRACK_ONLY`, and 6
`UNSUPPORTED` terminal states. The two overlap cells are REVE motor imagery and
mental arithmetic; they remain outside the 34-cell support-matched direct
surface. [E5, E6]

## Evidence and identity binding

| ID | Local evidence | Identity |
| --- | --- | --- |
| E1 | `comparative_methods/evidence/protected_campaign/joint_release_candidate_v1.json` | SHA-256 `806edc8d983efcd0ba87105b6b257eb39f05823d793f9b1f4a4f8984d0e2f78a` |
| E2 | Executed dual-GO authorization record, currently stored at the legacy path `comparative_methods/evidence/protected_campaign/authorization_template_v1.json` | SHA-256 `048a1ab2a3fab1d33e752a6dc7af367f0f51fd4a117aa020bddf3c0c64f3b2d1` |
| E3 | `comparative_methods/runs/protected_campaign/joint-comparison-protected-20260813-v3-single-gpu/joint-comparison-protected-20260813-v3-single-gpu/campaign_status.json` | SHA-256 `84eec4401d44624fdcfedf9c86f929935d1ecc8d312cf780f951730d04447eff` |
| E4 | `comparative_methods/evidence/protected_campaign/unblind_manifest_v1.json` | SHA-256 `1ae7ebcb51cb773f478cc2bd87cb674ed89f6dc1bf8f4f665c8f8d0f0ee86316` |
| E5 | `comparative_methods/runs/protected_campaign/joint-comparison-protected-20260813-v3-single-gpu/joint-comparison-protected-20260813-v3-single-gpu/aggregate/aggregate.json` | SHA-256 `7d521e2a1e1542dd208b4b737ab4c847f2d2f30228acd5d41013b1176427e961` |
| E6 | `comparative_methods/runs/protected_campaign/joint-comparison-protected-20260813-v3-single-gpu/joint-comparison-protected-20260813-v3-single-gpu/aggregate/cells.csv` | SHA-256 `a57981c4cd6677d8385f7560095a582adc8caf1465f476aa7c2f7b22406196b8` |
| E7 | `comparative_methods/runs/protected_campaign/joint-comparison-protected-20260813-v3-single-gpu/joint-comparison-protected-20260813-v3-single-gpu/aggregate/job_traceability.json` | SHA-256 `334664632b8bc5375e80813951162c0874a783f0f0bfaba8c95bdea859a1811a`; 540 rows |

The candidate, authorization, unblind, aggregate, and traceability records bind
the same campaign and candidate/authorization hashes. Local machine checks
confirmed the JSON identities and counts; accountable project owners remain
responsible for approving this report for external release.

## Joint 42-cell primary results

Values are the frozen primary endpoint mean and sample SD across five outer
folds. Seeds 17, 42, and 73 were averaged within each fold before the five-fold
summary. Classification values are macro-F1 on the 0–1 scale; REFED uses
native-coordinate masked CCC. A dash denotes a preregistered unsupported cell,
not a measured zero. [E5, E6]

| Method | Task | Track | Metric | Mean | Fold SD | Terminal |
| --- | --- | --- | --- | ---: | ---: | --- |
| BIOT | DSR | single-modal EEG official pretrained linear probe | macro-F1 | 0.4884 | 0.0671 | `REJECTED_VALUE` |
| BIOT | Mental arithmetic | single-modal EEG official pretrained linear probe | macro-F1 | 0.5677 | 0.0371 | `TABLE_READY_WITH_NOTE` |
| BIOT | Motor imagery | single-modal EEG official pretrained linear probe | macro-F1 | 0.5007 | 0.0170 | `REJECTED_VALUE` |
| BIOT | n-back | single-modal EEG official pretrained linear probe | macro-F1 | 0.3628 | 0.0325 | `TABLE_READY_WITH_NOTE` |
| BIOT | REFED | single-modal EEG official pretrained linear probe | masked CCC | — | — | `UNSUPPORTED` |
| BIOT | Visual | single-modal EEG official pretrained linear probe | macro-F1 | 0.2243 | 0.0124 | `REJECTED_VALUE` |
| BIOT | Word generation | single-modal EEG official pretrained linear probe | macro-F1 | 0.5781 | 0.0658 | `TABLE_READY_WITH_NOTE` |
| BrainFusion reimplementation | DSR | preregistered unsupported | macro-F1 | — | — | `UNSUPPORTED` |
| BrainFusion reimplementation | Mental arithmetic | cross-task supervised reimplementation | macro-F1 | 0.5502 | 0.0196 | `TABLE_READY_WITH_NOTE` |
| BrainFusion reimplementation | Motor imagery | source-case supervised reimplementation | macro-F1 | 0.5493 | 0.0090 | `TABLE_READY_WITH_NOTE` |
| BrainFusion reimplementation | n-back | cross-task supervised reimplementation | macro-F1 | 0.3337 | 0.0267 | `REJECTED_VALUE` |
| BrainFusion reimplementation | REFED | preregistered unsupported | masked CCC | — | — | `UNSUPPORTED` |
| BrainFusion reimplementation | Visual | cross-task supervised reimplementation | macro-F1 | 0.2222 | 0.0149 | `REJECTED_VALUE` |
| BrainFusion reimplementation | Word generation | cross-task supervised reimplementation | macro-F1 | 0.5428 | 0.0199 | `TABLE_READY_WITH_NOTE` |
| CBraMod | DSR | single-modal EEG official pretrained linear probe | macro-F1 | 0.5878 | 0.0363 | `TABLE_READY_WITH_NOTE` |
| CBraMod | Mental arithmetic | single-modal EEG official pretrained linear probe | macro-F1 | 0.6175 | 0.0286 | `TABLE_READY_WITH_NOTE` |
| CBraMod | Motor imagery | single-modal EEG official pretrained linear probe | macro-F1 | 0.5119 | 0.0195 | `REJECTED_VALUE` |
| CBraMod | n-back | single-modal EEG official pretrained linear probe | macro-F1 | 0.3880 | 0.0402 | `TABLE_READY_WITH_NOTE` |
| CBraMod | REFED | single-modal EEG official pretrained linear probe | masked CCC | — | — | `UNSUPPORTED` |
| CBraMod | Visual | single-modal EEG official pretrained linear probe | macro-F1 | 0.2238 | 0.0151 | `REJECTED_VALUE` |
| CBraMod | Word generation | single-modal EEG official pretrained linear probe | macro-F1 | 0.6100 | 0.0506 | `TABLE_READY_WITH_NOTE` |
| EFRM | DSR | multimodal target-dataset-excluded frozen linear probe | macro-F1 | 0.4156 | 0.0025 | `REJECTED_VALUE` |
| EFRM | Mental arithmetic | multimodal target-dataset-excluded frozen linear probe | macro-F1 | 0.6154 | 0.0764 | `TABLE_READY_WITH_NOTE` |
| EFRM | Motor imagery | multimodal target-dataset-excluded frozen linear probe | macro-F1 | 0.4748 | 0.0246 | `REJECTED_VALUE` |
| EFRM | n-back | multimodal target-dataset-excluded frozen linear probe | macro-F1 | 0.3697 | 0.0519 | `TABLE_READY_WITH_NOTE` |
| EFRM | REFED | multimodal target-dataset-excluded frozen linear probe | masked CCC | 0.0783 | 0.0104 | `TABLE_READY_WITH_NOTE` |
| EFRM | Visual | multimodal target-dataset-excluded frozen linear probe | macro-F1 | 0.1852 | 0.0340 | `REJECTED_VALUE` |
| EFRM | Word generation | multimodal target-dataset-excluded frozen linear probe | macro-F1 | 0.5406 | 0.0481 | `TABLE_READY_WITH_NOTE` |
| NormWear adapted | DSR | multimodal official-pretrained frozen linear probe, adapted | macro-F1 | 0.5800 | 0.0250 | `TABLE_READY_WITH_NOTE` |
| NormWear adapted | Mental arithmetic | multimodal official-pretrained frozen linear probe, adapted | macro-F1 | 0.6248 | 0.0344 | `TABLE_READY_WITH_NOTE` |
| NormWear adapted | Motor imagery | multimodal official-pretrained frozen linear probe, adapted | macro-F1 | 0.5414 | 0.0182 | `TABLE_READY_WITH_NOTE` |
| NormWear adapted | n-back | multimodal official-pretrained frozen linear probe, adapted | macro-F1 | 0.4167 | 0.0159 | `TABLE_READY_WITH_NOTE` |
| NormWear adapted | REFED | preregistered unsupported | masked CCC | — | — | `UNSUPPORTED` |
| NormWear adapted | Visual | multimodal official-pretrained frozen linear probe, adapted | macro-F1 | 0.2490 | 0.0297 | `REJECTED_VALUE` |
| NormWear adapted | Word generation | multimodal official-pretrained frozen linear probe, adapted | macro-F1 | 0.6125 | 0.0182 | `TABLE_READY_WITH_NOTE` |
| REVE | DSR | single-modal EEG official pretrained linear probe | macro-F1 | 0.6477 | 0.0521 | `TABLE_READY_WITH_NOTE` |
| REVE | Mental arithmetic | open-world pretrained with target-corpus overlap | macro-F1 | 0.5929 | 0.0207 | `OVERLAP_TRACK_ONLY` |
| REVE | Motor imagery | open-world pretrained with target-corpus overlap | macro-F1 | 0.5682 | 0.0125 | `OVERLAP_TRACK_ONLY` |
| REVE | n-back | single-modal EEG official pretrained linear probe | macro-F1 | 0.3926 | 0.0492 | `TABLE_READY_WITH_NOTE` |
| REVE | REFED | single-modal EEG official pretrained linear probe | masked CCC | — | — | `UNSUPPORTED` |
| REVE | Visual | single-modal EEG official pretrained linear probe | macro-F1 | 0.2408 | 0.0159 | `REJECTED_VALUE` |
| REVE | Word generation | single-modal EEG official pretrained linear probe | macro-F1 | 0.6183 | 0.0429 | `TABLE_READY_WITH_NOTE` |

## STA-Net context reference

STA-Net was not rerun in the 540-job joint campaign. Its previously frozen
strict cross-subject five-fold aggregate remains a
`method_native_context_reference` because its observation budget differs from
the support-matched direct surface. Classification values below are percentages;
REFED CCC is unitless. [E8]

| Task | Primary endpoint | Mean ± fold SD |
| --- | --- | ---: |
| Motor imagery | macro-F1 | 56.40 ± 1.58% |
| Mental arithmetic | macro-F1 | 62.84 ± 4.25% |
| Word generation | macro-F1 | 62.11 ± 3.13% |
| n-back | macro-F1 | 37.52 ± 2.32% |
| DSR | macro-F1 | 60.69 ± 2.38% |
| Visual | macro-F1 | 25.01 ± 0.77% |
| REFED | CCC | 0.081 ± 0.048 |

E8 is
`comparative_methods/STA-Net-PyTorch/runs/fivefold/20260727_sta_net_no_artifact_mask_converged_5fold_v1/aggregate/paper_table.md`.

## Interpretation and release boundaries

- `TABLE_READY_WITH_NOTE` may enter the appropriate same-track table with its
  required note; it is not an unqualified reproduction claim.
- `REJECTED_VALUE` is a valid observed result that failed the frozen numeric
  acceptance gate. It remains visible here and must not be silently dropped or
  represented as a normal admissible performance value.
- `OVERLAP_TRACK_ONLY` is valid only in the declared target-corpus-overlap table.
- `UNSUPPORTED` means the cell was excluded before protected evaluation and has
  no numeric result.
- Different tracks, observation budgets, modalities, and metrics must not be
  collapsed into a single overall method ranking.
- Raw protected predictions, per-job outputs, unblinding records, and full
  aggregates remain local under the repository's current release policy. This
  report is the tracked, human-readable result surface; it does not replace the
  local 540-row traceability record.
