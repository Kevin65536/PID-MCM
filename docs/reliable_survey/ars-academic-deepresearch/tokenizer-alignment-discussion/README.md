# Tokenizer Alignment Discussion — ARS Academic DeepResearch Archive

**Session Date**: 2026-06-24  
**Session ID**: `e0b2f6b5-82f7-41a0-b0fc-e3adb77916c7`  
**Original Transcript**: `~/.claude/projects/-SSD-2-pid-mcm-implementation/e0b2f6b5-82f7-41a0-b0fc-e3adb77916c7.jsonl`

## Context

This archive contains the papers referenced during an ARS (Academic Research Skills) full-pipeline deep-research session discussing the **neuro-tokenization coupling design problem**. The core research question was whether to:

1. **Explore information interaction design within the tokenizer** (training-stage coupling between EEG and fNIRS encoders), or
2. **Explore physiological coupling discovery via token sequence pretraining** (downstream-stage analysis of discrete token co-occurrence patterns)

## Archive Structure

```
tokenizer-alignment-discussion/
├── README.md           # This file
├── references.bib      # BibTeX bibliography (23 entries)
└── papers/             # Archived PDFs (23 unique papers, deduplicated from 53 WebFetch results)
    ├── baevski2020_wav2vec2.pdf
    ├── barmpas2026_neuroRVQ.pdf
    ├── cui2026_compact_latent_manifold.pdf
    ├── ...
    └── zhao2026_continuous_first.pdf
```

## Paper Categories

| Category | Count | Key Papers |
|----------|-------|-----------|
| **Vector Quantization / Codebook Methods** | 8 | Huh 2023 (STE fix), Mentzer 2023 (FSQ), Takida 2022 (SQ-VAE), Lu 2026 (PCA-VAE), Zhao 2026 (Continuous First) |
| **Cross-Modal Discrete Representation** | 4 | Liu 2021 (cross-modal VQ), Huang 2024 (semantic residual), Lei 2024 (M3-Jepa) |
| **Biosignal / Neuro Tokenization** | 3 | Barmpas 2026 (NeuroRVQ), Cui 2026 (EEG-fNIRS latent translation), Baevski 2020 (wav2vec 2.0) |
| **Causal Discovery / Hawkes Processes** | 5 | Qiao 2023 (structural Hawkes), Wu 2024 (Granger-Hawkes), Shou 2023, Idé 2019, Zennaro 2023 |
| **Gradient Estimation (discrete)** | 2 | Shekhovtsov 2021 (bias-variance), Shekhovtsov 2022 (Rao-Blackwellized STE) |
| **Outlier / Probably Irrelevant** | 1 | Joseph 2025 (plasma physics — arXiv ID collision) |

## Key Insights from the Session

The ARS deep-research workflow ran two major phases:

1. **Phase 1 (wf_f7a9a76e)**: Explored tokenizer architecture improvements — focusing on VQ training stability (STE fixes, FSQ, SQ-VAE, PCA-VAE), cross-modal discrete representation learning, and biosignal-specific tokenization (NeuroRVQ).

2. **Phase 2 (wf_d35d3ddd)**: Explored downstream causal discovery methods — focusing on Hawkes processes, Granger causality, and temporal point processes that could discover physiological coupling patterns from discrete token sequences.

## Related Project Experiments

The tokenizer experiments discussed in this session are located at:
- `experiments/runs/coupling_design_audit/`
- `experiments/runs/tokenizer_coupling_capacity/`
- `experiments/runs/tokenizer_coupling_discovery/`
- `experiments/runs/tokenizer_next_stage/`

## Notes

- 53 PDFs were originally fetched by sub-agents across 2 workflows (many duplicates due to parallel agent fan-out)
- 30 duplicates were removed, leaving 23 unique papers
- 1 irrelevant paper (plasma physics) was kept for archival completeness
- Some PDFs are arXiv preprints that may have been updated since the fetch date
- The `boz4ewf39.txt` HTML file (PCA-VAE title page) and `bys1edhu4.txt` (Structural Hawkes pdftotext) are text extractions stored in the tool-results but not re-archived here
