#!/usr/bin/env python
"""
Run PID-MCM experiment with configuration file.

Usage:
    python run_experiment.py --config E0_baseline.yaml
    python run_experiment.py --config E6_synergy_residual.yaml --epochs 50
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np

from src.utils.logger import ExperimentLogger, update_comparison_csv
from src.data.synthetic_timeseries import PIDTimeSeriesDataset, create_dataset_from_config


class SimpleELPEncoder(nn.Module):
    """
    Simplified ELP encoder for validation experiments.
    Uses FC layers instead of full Transformer for quick testing.
    """
    
    def __init__(self, seq_length: int = 256, hidden_dim: int = 128, token_dim: int = 32):
        super().__init__()
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.token_dim = token_dim
        
        # Shared encoder backbone
        self.encoder = nn.Sequential(
            nn.Linear(seq_length * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Token-specific projection heads
        self.head_r = nn.Linear(hidden_dim, token_dim)
        self.head_u1 = nn.Linear(hidden_dim, token_dim)
        self.head_u2 = nn.Linear(hidden_dim, token_dim)
        self.head_s = nn.Linear(hidden_dim, token_dim)
        
        # Decoder (shared)
        self.decoder = nn.Sequential(
            nn.Linear(token_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, seq_length * 2),
        )
    
    def encode(self, x1: torch.Tensor, x2: torch.Tensor):
        """Encode inputs to PID tokens."""
        x = torch.cat([x1, x2], dim=-1)  # [B, 2*seq]
        h = self.encoder(x)
        
        z_r = self.head_r(h)
        z_u1 = self.head_u1(h)
        z_u2 = self.head_u2(h)
        z_s = self.head_s(h)
        
        return z_r, z_u1, z_u2, z_s
    
    def decode(self, z_r, z_u1, z_u2, z_s):
        """Decode tokens back to observations."""
        z = torch.cat([z_r, z_u1, z_u2, z_s], dim=-1)
        out = self.decoder(z)
        x1_rec = out[:, :self.seq_length]
        x2_rec = out[:, self.seq_length:]
        return x1_rec, x2_rec
    
    def forward(self, x1, x2):
        z_r, z_u1, z_u2, z_s = self.encode(x1, x2)
        x1_rec, x2_rec = self.decode(z_r, z_u1, z_u2, z_s)
        return {
            'z_r': z_r, 'z_u1': z_u1, 'z_u2': z_u2, 'z_s': z_s,
            'x1_rec': x1_rec, 'x2_rec': x2_rec
        }


def compute_losses(outputs, batch, config):
    """Compute all loss components based on config."""
    x1, x2 = batch['x1'], batch['x2']
    z_r = outputs['z_r']
    z_u1 = outputs['z_u1']
    z_u2 = outputs['z_u2']
    z_s = outputs['z_s']
    x1_rec = outputs['x1_rec']
    x2_rec = outputs['x2_rec']
    
    loss_config = config.get('loss', {})
    
    # Reconstruction loss
    l_rec = nn.functional.mse_loss(x1_rec, x1) + nn.functional.mse_loss(x2_rec, x2)
    
    # Alignment loss (for z_r)
    align_type = loss_config.get('alignment', {}).get('type', 'A1_MSE')
    if align_type == 'A1_MSE':
        # In this simplified version, we just minimize variance across samples
        # as proxy for alignment (real impl would use different views)
        l_align = z_r.var(dim=0).mean()
    else:
        l_align = torch.tensor(0.0, device=x1.device)
    
    # Orthogonality loss
    orth_type = loss_config.get('orthogonality', {}).get('type', 'B1_Cosine')
    tokens = [z_r, z_u1, z_u2, z_s]
    l_orth = torch.tensor(0.0, device=x1.device)
    for i in range(len(tokens)):
        for j in range(i+1, len(tokens)):
            cos_sim = nn.functional.cosine_similarity(tokens[i], tokens[j], dim=-1)
            if orth_type == 'B1_Cosine':
                l_orth = l_orth + cos_sim.abs().mean()
            else:  # B2_SqCosine
                l_orth = l_orth + (cos_sim ** 2).mean()
    
    # Synergy loss
    syn_type = loss_config.get('synergy', {}).get('type', 'C1_MaskingDiff')
    if syn_type == 'C1_MaskingDiff':
        # Encourage z_s to have variance (not collapse)
        l_syn = -z_s.var(dim=0).mean()
    elif syn_type == 'C4_Residual':
        # z_s should help explain residual
        # Simplified: encourage z_s to be different from z_r + z_u
        combined = z_r + z_u1 + z_u2
        l_syn = -nn.functional.mse_loss(z_s, combined.detach())
    else:
        l_syn = torch.tensor(0.0, device=x1.device)
    
    # Weighted sum
    w_rec = loss_config.get('reconstruction', {}).get('weight', 1.0)
    w_align = loss_config.get('alignment', {}).get('weight', 0.5)
    w_orth = loss_config.get('orthogonality', {}).get('weight', 0.3)
    w_syn = loss_config.get('synergy', {}).get('weight', 0.2)
    
    total = w_rec * l_rec + w_align * l_align + w_orth * l_orth + w_syn * l_syn
    
    return {
        'total': total,
        'reconstruction': l_rec.item(),
        'alignment': l_align.item(),
        'orthogonality': l_orth.item(),
        'synergy': l_syn.item()
    }


def compute_metrics(model, dataset, device):
    """Compute latent recovery metrics on full dataset."""
    model.eval()
    
    all_z_r, all_z_u1, all_z_u2, all_z_s = [], [], [], []
    all_w_r, all_w_u1, all_w_u2, all_w_s = [], [], [], []
    
    with torch.no_grad():
        loader = DataLoader(dataset, batch_size=256, shuffle=False)
        for batch in loader:
            x1 = batch['x1'].to(device)
            x2 = batch['x2'].to(device)
            outputs = model(x1, x2)
            
            all_z_r.append(outputs['z_r'].cpu())
            all_z_u1.append(outputs['z_u1'].cpu())
            all_z_u2.append(outputs['z_u2'].cpu())
            all_z_s.append(outputs['z_s'].cpu())
            
            all_w_r.append(batch['w_r'])
            all_w_u1.append(batch['w_u1'])
            all_w_u2.append(batch['w_u2'])
            all_w_s.append(batch['w_s'])
    
    z_r = torch.cat(all_z_r, dim=0)
    z_u1 = torch.cat(all_z_u1, dim=0)
    z_s = torch.cat(all_z_s, dim=0)
    
    w_r = torch.cat(all_w_r, dim=0)
    w_u1 = torch.cat(all_w_u1, dim=0)
    w_s = torch.cat(all_w_s, dim=0)
    
    # Compute correlations (simplified: mean correlation across dimensions)
    def corr(a, b):
        # Use mean of each sample's features
        a_mean = a.mean(dim=-1)
        b_mean = b.mean(dim=-1)
        a_c = a_mean - a_mean.mean()
        b_c = b_mean - b_mean.mean()
        return (a_c * b_c).sum() / (torch.sqrt((a_c**2).sum()) * torch.sqrt((b_c**2).sum()) + 1e-8)
    
    metrics = {
        'corr_zr_wr': corr(z_r, w_r).item(),
        'corr_zu_wu': corr(z_u1, w_u1).item(),
        'corr_zs_ws': corr(z_s, w_s).item(),
        'std_zr': z_r.std().item(),
        'hsic_zr_zu': abs(nn.functional.cosine_similarity(z_r, z_u1, dim=-1).mean().item()),
    }
    
    model.train()
    return metrics


def run_experiment(config_path: str, epochs: Optional[int] = None):
    """Run a single experiment."""
    
    # Initialize logger
    logger = ExperimentLogger(config_path)
    config = logger.get_config()
    
    # Override epochs if specified
    if epochs is not None:
        config['training']['epochs'] = epochs
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] Using {device}")
    
    # Create dataset
    print("[Data] Creating synthetic time-series dataset...")
    dataset = create_dataset_from_config(config)
    
    # Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'],
        shuffle=True
    )
    
    # Create model
    print("[Model] Creating SimpleELPEncoder...")
    model = SimpleELPEncoder(
        seq_length=config['data']['params']['seq_length'],
        hidden_dim=config['model']['hidden_dim'],
        token_dim=config['model']['token_dim']
    ).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Training loop
    num_epochs = config['training']['epochs']
    print(f"[Training] Starting {num_epochs} epochs...")
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_losses = []
        
        for batch in train_loader:
            x1 = batch['x1'].to(device)
            x2 = batch['x2'].to(device)
            batch_device = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                          for k, v in batch.items()}
            
            optimizer.zero_grad()
            outputs = model(x1, x2)
            losses = compute_losses(outputs, batch_device, config)
            
            losses['total'].backward()
            optimizer.step()
            
            epoch_losses.append({k: v for k, v in losses.items() if k != 'total'})
        
        # Average losses
        avg_losses = {k: np.mean([l[k] for l in epoch_losses]) for k in epoch_losses[0]}
        train_loss = np.mean([l['reconstruction'] for l in epoch_losses])
        
        # Compute metrics every N epochs
        if epoch % config['logging']['log_every_n_epochs'] == 0:
            metrics = compute_metrics(model, dataset, device)
            logger.log_epoch(
                epoch=epoch,
                train_loss=train_loss,
                loss_breakdown=avg_losses,
                metrics=metrics
            )
        else:
            logger.log_epoch(epoch=epoch, train_loss=train_loss)
        
        # Save checkpoint
        if epoch % config['logging']['save_checkpoint_every'] == 0:
            logger.save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, epoch)
    
    # Final evaluation
    print("[Evaluation] Computing final metrics...")
    final_metrics = compute_metrics(model, dataset, device)
    logger.log_final(final_metrics)
    
    # Generate figures
    if config['logging']['generate_figures']:
        logger.generate_figures()
        
        # Generate enhanced analysis figures
        print("[Analysis] Generating signal reconstruction and PID analysis...")
        model.eval()
        with torch.no_grad():
            # Get a batch for visualization
            sample_batch = next(iter(DataLoader(dataset, batch_size=min(100, len(dataset)), shuffle=False)))
            x1 = sample_batch['x1'].to(device)
            x2 = sample_batch['x2'].to(device)
            outputs = model(x1, x2)
            
            # Signal reconstruction plot
            logger.plot_signal_reconstruction(
                x1_orig=sample_batch['x1'].numpy(),
                x2_orig=sample_batch['x2'].numpy(),
                x1_rec=outputs['x1_rec'].cpu().numpy(),
                x2_rec=outputs['x2_rec'].cpu().numpy(),
                sample_idx=0,
                fs=config['data']['params']['fs']
            )
            
            # Collect all tokens for PID analysis
            all_z_r, all_z_u1, all_z_u2, all_z_s = [], [], [], []
            all_w_r, all_w_u1, all_w_u2, all_w_s = [], [], [], []
            
            for batch in DataLoader(dataset, batch_size=256, shuffle=False):
                x1_b = batch['x1'].to(device)
                x2_b = batch['x2'].to(device)
                out = model(x1_b, x2_b)
                
                all_z_r.append(out['z_r'].cpu().numpy())
                all_z_u1.append(out['z_u1'].cpu().numpy())
                all_z_u2.append(out['z_u2'].cpu().numpy())
                all_z_s.append(out['z_s'].cpu().numpy())
                
                all_w_r.append(batch['w_r'].numpy())
                all_w_u1.append(batch['w_u1'].numpy())
                all_w_u2.append(batch['w_u2'].numpy())
                all_w_s.append(batch['w_s'].numpy())
            
            # PID analysis plot
            logger.plot_pid_analysis(
                z_r=np.concatenate(all_z_r, axis=0),
                z_u1=np.concatenate(all_z_u1, axis=0),
                z_u2=np.concatenate(all_z_u2, axis=0),
                z_s=np.concatenate(all_z_s, axis=0),
                w_r=np.concatenate(all_w_r, axis=0),
                w_u1=np.concatenate(all_w_u1, axis=0),
                w_u2=np.concatenate(all_w_u2, axis=0),
                w_s=np.concatenate(all_w_s, axis=0),
                fs=config['data']['params']['fs']
            )
    
    # Update comparison CSV
    update_comparison_csv()
    
    print(f"\n[Done] Experiment completed: {logger.run_name}")
    print(f"[Results] Final metrics:")
    for k, v in final_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    return logger.run_name, final_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PID-MCM experiment")
    parser.add_argument("--config", type=str, required=True, help="Config file name")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    
    args = parser.parse_args()
    run_experiment(args.config, args.epochs)
