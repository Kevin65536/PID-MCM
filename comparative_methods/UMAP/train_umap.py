"""
UMAP Training Script — Pretraining & Finetuning

Usage:
    # Pretrain
    python train_umap.py pretrain --config configs/pretrain.yaml

    # Finetune (multimodal)
    python train_umap.py finetune --config configs/finetune.yaml

    # Finetune (missing fNIRS = EEG only)
    python train_umap.py finetune --config configs/finetune.yaml --modality eeg

    # Smoke test
    python train_umap.py pretrain --config configs/pretrain.yaml --epochs 3 --run_name smoke
"""

import os
import sys
import json
import time
import argparse
import datetime
from pathlib import Path
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from umap_dataset import UMAPDataset, create_umap_dataloaders, collate_missing_modality
from model import UMAPPretrain, UMAPFinetune
from model.umap_utils import adjust_learning_rate, compute_acc


# ===========================================================================
# Config loading
# ===========================================================================

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """Override config values from CLI flags (if not None)."""
    overrides = {
        'training.epochs': args.epochs,
        'training.batch_size': args.batch_size,
        'training.lr': args.lr,
        'data.task': args.task,
        'data.seq_length': args.seq_length,
        'data.feature_mode': args.feature_mode,
        'logging.run_name': args.run_name,
    }
    if hasattr(args, 'modality') and args.modality is not None:
        overrides['model.modality'] = args.modality
    if hasattr(args, 'pretrain_ckpt') and args.pretrain_ckpt is not None:
        overrides['pretrain.checkpoint'] = args.pretrain_ckpt

    for dotpath, val in overrides.items():
        if val is None:
            continue
        keys = dotpath.split('.')
        d = cfg
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = val

    return cfg


# ===========================================================================
# Experiment Logger
# ===========================================================================

class ExperimentLogger:
    """Logging, checkpointing, and plotting for UMAP experiments."""

    def __init__(self, run_dir: Path, config: dict):
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = run_dir / 'checkpoints'
        self.plot_dir = run_dir / 'plots'
        self.ckpt_dir.mkdir(exist_ok=True)
        self.plot_dir.mkdir(exist_ok=True)

        with open(run_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2, default=str)

        self.history = {'train': {}, 'val': {}, 'test': {}}
        self.best_val_metric = None
        self.best_val_epoch = -1
        self.log_file = open(run_dir / 'training.log', 'w')
        self.log(f"Experiment started at {datetime.datetime.now()}")

    def log(self, msg: str):
        ts = datetime.datetime.now().strftime('%H:%M:%S')
        line = f"[{ts}] {msg}"
        print(line)
        self.log_file.write(line + '\n')
        self.log_file.flush()

    def log_epoch(self, epoch: int, phase: str, metrics: dict):
        for k, v in metrics.items():
            self.history[phase].setdefault(k, []).append(v)
        s = ', '.join(f'{k}={v:.6f}' for k, v in metrics.items())
        self.log(f"Epoch {epoch} [{phase}] {s}")

    def update_best(self, epoch, val_metric, model, metric_name='val_metric', mode='min'):
        is_best = (
            self.best_val_metric is None
            or (mode == 'min' and val_metric < self.best_val_metric)
            or (mode == 'max' and val_metric > self.best_val_metric)
        )
        if is_best:
            self.best_val_metric = val_metric
            self.best_val_epoch = epoch
            self.save_checkpoint(model, 'best_checkpoint.pth', epoch)
            self.log(f"  ★ New best {metric_name}: {val_metric:.6f} @ epoch {epoch}")
        return is_best

    def save_checkpoint(self, model, filename, epoch, optimizer=None):
        ckpt = {'epoch': epoch, 'model': model.state_dict()}
        if optimizer:
            ckpt['optimizer'] = optimizer.state_dict()
        torch.save(ckpt, self.ckpt_dir / filename)

    def save_periodic(self, model, optimizer, epoch, every_n=20):
        if (epoch + 1) % every_n == 0:
            self.save_checkpoint(model, f'checkpoint-{epoch}.pth', epoch, optimizer)

    def plot_curves(self, keys, title='Loss', filename='curves.png'):
        fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4))
        if len(keys) == 1:
            axes = [axes]
        for ax, k in zip(axes, keys):
            if k in self.history['train']:
                ax.plot(self.history['train'][k], label='train', color='#2196F3')
            if k in self.history['val'] and self.history['val'][k]:
                ax.plot(self.history['val'][k], label='val', color='#FF5722', linestyle='--')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(k)
            ax.set_title(k)
            ax.legend()
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.plot_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()

    def save_results(self, results: dict):
        with open(self.run_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

    def finalize(self):
        # Save history
        h = {p: {k: [float(v) for v in vs] for k, vs in d.items()} for p, d in self.history.items()}
        with open(self.run_dir / 'history.json', 'w') as f:
            json.dump(h, f, indent=2)
        self.log(f"Best val: {self.best_val_metric} @ epoch {self.best_val_epoch}")
        self.log_file.close()


# ===========================================================================
# Model factory
# ===========================================================================

def _make_qformer_config(model_cfg: dict):
    from transformers import Blip2QFormerConfig
    return Blip2QFormerConfig(
        hidden_size=model_cfg.get('hidden_size', 64),
        intermediate_size=model_cfg.get('intermediate_size', 256),
        num_attention_heads=model_cfg.get('num_attention_heads', 4),
        num_hidden_layers=model_cfg.get('num_hidden_layers', 4),
        hidden_act=model_cfg.get('hidden_act', 'gelu'),
        hidden_dropout_prob=model_cfg.get('hidden_dropout_prob', 0.1),
        attention_probs_dropout_prob=model_cfg.get('attention_probs_dropout_prob', 0.1),
        initializer_range=model_cfg.get('initializer_range', 0.02),
        layer_norm_eps=model_cfg.get('layer_norm_eps', 1e-12),
        chunk_size_feed_forward=model_cfg.get('chunk_size_feed_forward', 0),
        output_hidden_states=model_cfg.get('output_hidden_states', True),
        if_DDP=model_cfg.get('if_DDP', False),
    )


def create_pretrain_model(dataset_info: dict, model_cfg: dict, device='cuda'):
    config = _make_qformer_config(model_cfg)
    config.if_DDP = model_cfg.get('if_DDP', False)
    model = UMAPPretrain(
        umap_config=config,
        umap_device=torch.device(device),
        seq_length=dataset_info['seq_length'],
        eeg_input_dim=dataset_info['eeg_input_dim'],
        eye_input_dim=dataset_info['fnirs_input_dim'],
    ).to(device)
    return model, config


def create_finetune_model(dataset_info: dict, model_cfg: dict, device='cuda'):
    config = _make_qformer_config(model_cfg)
    model = UMAPFinetune(
        umap_config=config,
        umap_device=torch.device(device),
        seq_length=dataset_info['seq_length'],
        eeg_input_dim=dataset_info['eeg_input_dim'],
        eye_input_dim=dataset_info['fnirs_input_dim'],
        n_class=model_cfg.get('n_class', 2),
        mode=model_cfg.get('modality', 'multi'),
    ).to(device)
    return model, config


def load_pretrain_weights(model, checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state = model.state_dict()
    loaded = 0
    for k in state:
        if k in ckpt['model']:
            state[k] = ckpt['model'][k]
            loaded += 1
    model.load_state_dict(state)
    print(f"Loaded {loaded}/{len(state)} params from pretrain checkpoint")
    return model


# ===========================================================================
# Training loops
# ===========================================================================

def pretrain_one_epoch(model, loader, optimizer, device, epoch, logger, ablation=None):
    model.train()
    sums = {'loss_total': 0, 'loss_con': 0, 'loss_mat': 0, 'lm_loss': 0}
    n = 0
    for batch in loader:
        eeg = batch['eeg'].to(device)
        fnirs = batch['fnirs'].to(device)
        loss, loss_con, loss_mat, lm_loss = model(eeg, fnirs)

        if ablation is not None:
            loss = sum(v for k, v in [('con', loss_con), ('mat', loss_mat), ('gen', lm_loss)] if k in ablation)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        sums['loss_total'] += loss.item()
        sums['loss_con'] += loss_con.item()
        sums['loss_mat'] += loss_mat.item()
        sums['lm_loss'] += lm_loss.item()
        n += 1

    metrics = {k: v / n for k, v in sums.items()}
    logger.log_epoch(epoch, 'train', metrics)
    return metrics


@torch.no_grad()
def pretrain_eval(model, loader, device, epoch, logger, phase='val'):
    model.eval()
    sums = {'loss_total': 0, 'loss_con': 0, 'loss_mat': 0, 'lm_loss': 0}
    n = 0
    for batch in loader:
        eeg = batch['eeg'].to(device)
        fnirs = batch['fnirs'].to(device)
        loss, loss_con, loss_mat, lm_loss = model(eeg, fnirs)
        sums['loss_total'] += loss.item()
        sums['loss_con'] += loss_con.item()
        sums['loss_mat'] += loss_mat.item()
        sums['lm_loss'] += lm_loss.item()
        n += 1
    metrics = {k: v / n for k, v in sums.items()}
    logger.log_epoch(epoch, phase, metrics)
    return metrics


def finetune_one_epoch(model, loader, optimizer, device, epoch, logger):
    model.train()
    total_loss, correct, total = 0, 0, 0
    n = 0
    for batch in loader:
        eeg = batch['eeg'].to(device) if batch['eeg'] is not None else None
        fnirs = batch['fnirs'].to(device) if batch['fnirs'] is not None else None
        labels = batch['label'].to(device)

        logits = model(eeg=eeg, eye=fnirs)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
        n += 1

    metrics = {'loss': total_loss / n, 'accuracy': correct / total}
    logger.log_epoch(epoch, 'train', metrics)
    return metrics


@torch.no_grad()
def finetune_eval(model, loader, device, epoch, logger, phase='val'):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    subj_correct, subj_total = {}, {}
    n = 0

    for batch in loader:
        eeg = batch['eeg'].to(device) if batch['eeg'] is not None else None
        fnirs = batch['fnirs'].to(device) if batch['fnirs'] is not None else None
        labels = batch['label'].to(device)
        subjects = batch['subject_id']

        logits = model(eeg=eeg, eye=fnirs)
        loss = F.cross_entropy(logits, labels)
        preds = logits.argmax(1)

        total_loss += loss.item()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        for i, s in enumerate(subjects):
            s = int(s)
            subj_correct[s] = subj_correct.get(s, 0) + int(preds[i] == labels[i])
            subj_total[s] = subj_total.get(s, 0) + 1
        n += 1

    metrics = {'loss': total_loss / n, 'accuracy': correct / total}
    logger.log_epoch(epoch, phase, metrics)
    subj_accs = {s: subj_correct[s] / subj_total[s] for s in subj_correct}
    return metrics, np.array(all_preds), np.array(all_labels), subj_accs


# ===========================================================================
# Main runners
# ===========================================================================

def run_pretrain(cfg: dict):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dcfg = cfg['data']
    tcfg = cfg['training']

    dataloaders, dinfo = create_umap_dataloaders(
        data_root=str(PROJECT_ROOT / dcfg['root']),
        task=dcfg['task'],
        seq_length=dcfg['seq_length'],
        window_duration_s=dcfg.get('window_duration_s', 10.0),
        feature_mode=dcfg['feature_mode'],
        batch_size=tcfg['batch_size'],
        train_subjects=dcfg.get('train_subjects'),
        val_subjects=dcfg.get('val_subjects'),
        test_subjects=dcfg.get('test_subjects'),
    )

    model, qf_cfg = create_pretrain_model(dinfo, cfg['model'], device=str(device))
    n_params = sum(p.numel() for p in model.parameters())

    run_name = cfg['logging'].get('run_name') or f"UMAP_PT_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = SCRIPT_DIR / 'runs' / run_name

    exp_cfg = {**cfg, 'dataset_info': dinfo, 'n_params': n_params, 'device': str(device)}
    logger = ExperimentLogger(run_dir, exp_cfg)
    logger.log(f"Parameters: {n_params:,}")
    logger.log(f"Train: {dinfo['n_train']} samples | EEG dim={dinfo['eeg_input_dim']} | fNIRS dim={dinfo['fnirs_input_dim']}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg['lr'], weight_decay=tcfg['weight_decay'])

    ablation = set(tcfg['ablation_tasks'].split(',')) if tcfg.get('ablation_tasks') else None
    ckpt_every = cfg['logging'].get('checkpoint_every_n', 20)
    plot_every = cfg['logging'].get('plot_every_n', 10)

    for epoch in range(tcfg['epochs']):
        lr = adjust_learning_rate(
            optimizer, epoch,
            argparse.Namespace(lr=tcfg['lr'], min_lr=tcfg.get('min_lr', 1e-6),
                               warmup_epochs=tcfg.get('warmup_epochs', 10),
                               epochs=tcfg['epochs'])
        )
        pretrain_one_epoch(model, dataloaders['train'], optimizer, device, epoch, logger, ablation)
        val_m = pretrain_eval(model, dataloaders['val'], device, epoch, logger, 'val')
        logger.update_best(epoch, val_m['loss_total'], model, 'val_loss', mode='min')
        logger.save_periodic(model, optimizer, epoch, ckpt_every)

        if (epoch + 1) % plot_every == 0:
            logger.plot_curves(['loss_total', 'loss_con', 'loss_mat', 'lm_loss'],
                               title='Pretrain', filename='pretrain_curves.png')

    logger.plot_curves(['loss_total', 'loss_con', 'loss_mat', 'lm_loss'],
                       title='Pretrain', filename='pretrain_curves.png')
    logger.save_results({
        'best_val_loss': logger.best_val_metric,
        'best_epoch': logger.best_val_epoch,
        'final_train': {k: v[-1] for k, v in logger.history['train'].items()},
        'final_val': {k: v[-1] for k, v in logger.history['val'].items()},
    })
    logger.finalize()
    return run_dir


def run_finetune(cfg: dict):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dcfg = cfg['data']
    tcfg = cfg['training']
    mcfg = cfg['model']
    modality = mcfg.get('modality', 'multi')

    dataloaders, dinfo = create_umap_dataloaders(
        data_root=str(PROJECT_ROOT / dcfg['root']),
        task=dcfg['task'],
        seq_length=dcfg['seq_length'],
        window_duration_s=dcfg.get('window_duration_s', 10.0),
        feature_mode=dcfg['feature_mode'],
        batch_size=tcfg['batch_size'],
        train_subjects=dcfg.get('train_subjects'),
        val_subjects=dcfg.get('val_subjects'),
        test_subjects=dcfg.get('test_subjects'),
    )

    model, qf_cfg = create_finetune_model(dinfo, mcfg, device=str(device))
    n_params = sum(p.numel() for p in model.parameters())

    # Load pretrain weights
    pt_ckpt = cfg.get('pretrain', {}).get('checkpoint')
    if pt_ckpt:
        model = load_pretrain_weights(model, pt_ckpt)

    run_name = cfg['logging'].get('run_name') or f"UMAP_FT_{modality}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = SCRIPT_DIR / 'runs' / run_name

    exp_cfg = {**cfg, 'dataset_info': dinfo, 'n_params': n_params, 'device': str(device)}
    logger = ExperimentLogger(run_dir, exp_cfg)
    logger.log(f"Parameters: {n_params:,} | Modality: {modality} | Pretrained: {bool(pt_ckpt)}")

    # Re-wrap dataloaders with missing-modality collate
    collate_fn = partial(collate_missing_modality, mode=modality)
    for split in dataloaders:
        dl = dataloaders[split]
        dataloaders[split] = DataLoader(
            dl.dataset, batch_size=tcfg['batch_size'],
            shuffle=(split == 'train'), num_workers=0,
            pin_memory=True, drop_last=(split == 'train'),
            collate_fn=collate_fn,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg['lr'], weight_decay=tcfg['weight_decay'])
    ckpt_every = cfg['logging'].get('checkpoint_every_n', 20)
    plot_every = cfg['logging'].get('plot_every_n', 10)
    patience = tcfg.get('early_stopping', {}).get('patience', 999)
    no_improve = 0

    for epoch in range(tcfg['epochs']):
        finetune_one_epoch(model, dataloaders['train'], optimizer, device, epoch, logger)
        val_m, _, _, _ = finetune_eval(model, dataloaders['val'], device, epoch, logger, 'val')

        is_best = logger.update_best(epoch, val_m['accuracy'], model, 'val_accuracy', mode='max')
        if is_best:
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.log(f"Early stopping triggered at epoch {epoch}")
                break

        logger.save_periodic(model, optimizer, epoch, ckpt_every)
        if (epoch + 1) % plot_every == 0:
            logger.plot_curves(['loss', 'accuracy'], title='Finetune', filename='finetune_curves.png')

    # Final test evaluation with best model
    best_ckpt = run_dir / 'checkpoints' / 'best_checkpoint.pth'
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt['model'])
    test_m, test_preds, test_labels, subj_accs = finetune_eval(
        model, dataloaders['test'], device, -1, logger, 'test'
    )

    # Confusion matrix & report
    results = {
        'best_val_accuracy': logger.best_val_metric,
        'best_val_epoch': logger.best_val_epoch,
        'test_accuracy': test_m['accuracy'],
        'test_loss': test_m['loss'],
        'subject_accs': {str(k): v for k, v in subj_accs.items()},
    }
    if len(test_preds) > 0:
        try:
            from sklearn.metrics import classification_report, f1_score
            f1 = f1_score(test_labels, test_preds, average='macro')
            results['test_f1_macro'] = float(f1)
            results['classification_report'] = classification_report(
                test_labels, test_preds, output_dict=True
            )
        except ImportError:
            pass

    logger.plot_curves(['loss', 'accuracy'], title='Finetune', filename='finetune_curves.png')
    logger.save_results(results)
    logger.finalize()
    return run_dir


# ===========================================================================
# CLI
# ===========================================================================

def build_parser():
    p = argparse.ArgumentParser(description='UMAP Training')
    sub = p.add_subparsers(dest='command')

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--config', type=str, required=True, help='YAML config file')
    common.add_argument('--epochs', type=int, default=None)
    common.add_argument('--batch_size', type=int, default=None)
    common.add_argument('--lr', type=float, default=None)
    common.add_argument('--task', type=str, default=None)
    common.add_argument('--seq_length', type=int, default=None)
    common.add_argument('--feature_mode', type=str, default=None)
    common.add_argument('--run_name', type=str, default=None)

    sub.add_parser('pretrain', parents=[common])

    ft = sub.add_parser('finetune', parents=[common])
    ft.add_argument('--modality', type=str, default=None, choices=['multi', 'eeg', 'eye'])
    ft.add_argument('--pretrain_ckpt', type=str, default=None)

    return p


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)

    if args.command == 'pretrain':
        d = run_pretrain(cfg)
        print(f"\nPretrain complete → {d}")
    elif args.command == 'finetune':
        d = run_finetune(cfg)
        print(f"\nFinetune complete → {d}")
