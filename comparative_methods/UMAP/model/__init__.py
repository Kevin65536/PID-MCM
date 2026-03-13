"""
UMAP Model Package — Original model code from:
  "Multimodal Emotion Recognition with Missing Modality via 
   A Unified Multi-task Pre-training Framework"

Files:
  umap_qformer.py  — Core UMAP Transformer (QFormer-based encoder)
  umap_pretrain.py — Pretraining wrapper (contrastive + matching + generation)
  umap_finetune.py — Finetuning wrapper (classification with missing modality)
  umap_utils.py    — Utility functions (LR schedule, metric logger, DDP)
  config.py        — Config loader (YAML → Dict)
"""

from .umap_qformer import UMAP, UMAPEncoder, UMAPPreTrainedModel
from .umap_pretrain import UMAPPretrain
from .umap_finetune import UMAPFinetune
