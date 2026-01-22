"""
TensorBoard Logger for Tokenizer Training.

Provides comprehensive logging including:
1. Scalar metrics (loss, utilization, etc.)
2. Signal reconstruction plots
3. Codebook usage histograms
4. t-SNE/PCA embedding visualizations
5. Spectral analysis plots
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import io

# Matplotlib setup
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    SummaryWriter = None

# Sklearn for dimensionality reduction
try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def fig_to_image(fig) -> np.ndarray:
    """Convert matplotlib figure to numpy array for TensorBoard."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    
    from PIL import Image
    img = Image.open(buf)
    img_array = np.array(img)
    buf.close()
    
    # Handle RGBA -> RGB
    if img_array.ndim == 3 and img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
    
    return img_array


class TensorBoardLogger:
    """
    TensorBoard Logger for tokenizer training.
    
    Usage:
        tb_logger = TensorBoardLogger(run_dir)
        
        # Log scalars
        tb_logger.log_scalars("train", {"loss": 0.5, "vq_loss": 0.1}, step=1)
        
        # Log reconstruction (periodically)
        tb_logger.log_reconstruction(original, reconstructed, step=10)
        
        # Log embedding visualization
        tb_logger.log_embedding_tsne(embeddings, usage, step=10)
    """
    
    def __init__(
        self,
        run_dir: Union[str, Path],
        log_subdir: str = "tensorboard",
        comment: str = ""
    ):
        """
        Initialize TensorBoard logger.
        
        Args:
            run_dir: Experiment run directory
            log_subdir: Subdirectory for TensorBoard logs
            comment: Optional comment for the run (not used to avoid subdirectories)
        """
        if not HAS_TENSORBOARD:
            print("[TensorBoard] Warning: TensorBoard not available. Install with: pip install tensorboard")
            self.writer = None
            return
        
        self.run_dir = Path(run_dir)
        self.log_dir = self.run_dir / log_subdir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Use log_dir directly without comment to avoid creating subdirectories
        # This ensures all events go into a single file per run
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
        # Color palette for plots
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'tertiary': '#F18F01',
            'success': '#2ECC71',
            'danger': '#E74C3C',
        }
        
        print(f"[TensorBoard] Initialized at: {self.log_dir}")
    
    @property
    def enabled(self) -> bool:
        """Check if TensorBoard is enabled."""
        return self.writer is not None
    
    def close(self):
        """Close the TensorBoard writer."""
        if self.writer:
            self.writer.close()
    
    # =========================================================================
    # Scalar Logging
    # =========================================================================
    
    def log_scalars(
        self,
        tag_prefix: str,
        scalars: Dict[str, float],
        step: int
    ):
        """
        Log multiple scalars with a common prefix.
        
        Args:
            tag_prefix: Prefix for tags (e.g., "train", "val")
            scalars: Dict of scalar name -> value
            step: Global step (epoch or iteration)
        """
        if not self.enabled:
            return
        
        for name, value in scalars.items():
            if value is not None and not np.isnan(value):
                self.writer.add_scalar(f"{tag_prefix}/{name}", value, step)
    
    def log_learning_rate(self, lr: float, step: int):
        """Log learning rate."""
        if self.enabled:
            self.writer.add_scalar("train/learning_rate", lr, step)
    
    # =========================================================================
    # Reconstruction Visualization
    # =========================================================================
    
    def log_reconstruction(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        step: int,
        n_samples: int = 4,
        fs: float = 200.0,
        tag: str = "reconstruction"
    ):
        """
        Log reconstruction comparison plot.
        
        Args:
            original: Original signals [B, T] or [B, C, T]
            reconstructed: Reconstructed signals (same shape)
            step: Global step
            n_samples: Number of samples to display
            fs: Sampling frequency
            tag: Tag for the image
        """
        if not self.enabled:
            return
        
        # Convert to numpy
        if isinstance(original, torch.Tensor):
            original = original.detach().cpu().numpy()
        if isinstance(reconstructed, torch.Tensor):
            reconstructed = reconstructed.detach().cpu().numpy()
        
        # Handle 3D input
        if original.ndim == 3:
            original = original[:, 0, :]  # First channel
            reconstructed = reconstructed[:, 0, :]
        
        n_samples = min(n_samples, len(original))
        
        fig, axes = plt.subplots(n_samples, 2, figsize=(12, 2.5 * n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(n_samples):
            orig = original[i]
            recon = reconstructed[i]
            t = np.arange(len(orig)) / fs
            
            # Time domain
            ax_time = axes[i, 0]
            ax_time.plot(t, orig, label='Original', color=self.colors['primary'], 
                        alpha=0.8, linewidth=1.2)
            ax_time.plot(t, recon, label='Reconstructed', color=self.colors['secondary'],
                        alpha=0.8, linewidth=1.2, linestyle='--')
            ax_time.set_xlabel('Time (s)')
            ax_time.set_ylabel('Amplitude')
            ax_time.set_title(f'Sample {i+1}: Time Domain')
            ax_time.legend(loc='upper right', fontsize=8)
            ax_time.grid(True, alpha=0.3)
            
            # MSE annotation
            mse = np.mean((orig - recon) ** 2)
            ax_time.text(0.02, 0.98, f'MSE: {mse:.4f}', transform=ax_time.transAxes,
                        verticalalignment='top', fontsize=9,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Residual
            ax_res = axes[i, 1]
            residual = orig - recon
            ax_res.plot(t, residual, color=self.colors['danger'], alpha=0.7, linewidth=1)
            ax_res.fill_between(t, residual, alpha=0.3, color=self.colors['danger'])
            ax_res.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax_res.set_xlabel('Time (s)')
            ax_res.set_ylabel('Residual')
            ax_res.set_title(f'Sample {i+1}: Error')
            ax_res.grid(True, alpha=0.3)
        
        fig.suptitle(f'Reconstruction Quality (Step {step})', fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        # Convert to image and log
        img = fig_to_image(fig)
        self.writer.add_image(f"images/{tag}", img, step, dataformats='HWC')
        plt.close(fig)
    
    def log_spectral_comparison(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        step: int,
        fs: float = 200.0,
        n_samples: int = 100,
        tag: str = "spectral"
    ):
        """
        Log spectral comparison (PSD) plot.
        
        Args:
            original: Original signals
            reconstructed: Reconstructed signals
            step: Global step
            fs: Sampling frequency
            n_samples: Number of samples to average
            tag: Tag for the image
        """
        if not self.enabled:
            return
        
        # Convert to numpy
        if isinstance(original, torch.Tensor):
            original = original.detach().cpu().numpy()
        if isinstance(reconstructed, torch.Tensor):
            reconstructed = reconstructed.detach().cpu().numpy()
        
        if original.ndim == 3:
            original = original[:, 0, :]
            reconstructed = reconstructed[:, 0, :]
        
        n_use = min(n_samples, len(original))
        original = original[:n_use]
        reconstructed = reconstructed[:n_use]
        
        # Simple FFT-based PSD
        def compute_avg_psd(signals):
            psds = []
            for sig in signals:
                fft = np.fft.rfft(sig)
                psd = np.abs(fft) ** 2
                psds.append(psd)
            freqs = np.fft.rfftfreq(len(signals[0]), 1/fs)
            return freqs, np.mean(psds, axis=0)
        
        freq, psd_orig = compute_avg_psd(original)
        _, psd_recon = compute_avg_psd(reconstructed)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # PSD Comparison
        ax1 = axes[0]
        ax1.semilogy(freq, psd_orig + 1e-10, label='Original', 
                    color=self.colors['primary'], linewidth=2)
        ax1.semilogy(freq, psd_recon + 1e-10, label='Reconstructed', 
                    color=self.colors['secondary'], linewidth=2, linestyle='--')
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Power Spectral Density')
        ax1.set_title('PSD Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, fs/2])
        
        # PSD Ratio
        ax2 = axes[1]
        ratio = (psd_recon + 1e-10) / (psd_orig + 1e-10)
        ax2.plot(freq, ratio, color=self.colors['tertiary'], linewidth=2)
        ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=1)
        ax2.fill_between(freq, 0.8, 1.2, alpha=0.2, color=self.colors['success'])
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Ratio (Recon/Orig)')
        ax2.set_title('Spectral Fidelity')
        ax2.set_ylim([0, 2])
        ax2.set_xlim([0, fs/2])
        ax2.grid(True, alpha=0.3)
        
        fig.suptitle(f'Spectral Analysis (Step {step})', fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        img = fig_to_image(fig)
        self.writer.add_image(f"images/{tag}", img, step, dataformats='HWC')
        plt.close(fig)
    
    # =========================================================================
    # Codebook Analysis
    # =========================================================================
    
    def log_codebook_usage(
        self,
        indices: torch.Tensor,
        codebook_size: int,
        step: int,
        tag: str = "codebook"
    ):
        """
        Log codebook usage histogram and statistics.
        
        Args:
            indices: Token indices [B, T] or flattened
            codebook_size: Total codebook size
            step: Global step
            tag: Tag for the image
        """
        if not self.enabled:
            return
        
        # Compute usage
        if isinstance(indices, torch.Tensor):
            indices = indices.detach().cpu()
            flat = indices.flatten().long()
            usage = torch.bincount(flat, minlength=codebook_size).numpy()
        else:
            flat = np.array(indices).flatten()
            usage = np.bincount(flat.astype(int), minlength=codebook_size)
        
        # Statistics
        used_codes = (usage > 0).sum()
        dead_codes = codebook_size - used_codes
        utilization = used_codes / codebook_size
        
        # Entropy and perplexity
        total = usage.sum()
        probs = usage / (total + 1e-10)
        probs_nonzero = probs[probs > 0]
        entropy = -np.sum(probs_nonzero * np.log(probs_nonzero))
        perplexity = np.exp(entropy)
        
        # Log scalar metrics
        self.writer.add_scalar(f"{tag}/utilization", utilization, step)
        self.writer.add_scalar(f"{tag}/dead_codes", dead_codes, step)
        self.writer.add_scalar(f"{tag}/perplexity", perplexity, step)
        
        # Create histogram figure
        sorted_usage = np.sort(usage)[::-1]
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Usage histogram
        ax1 = axes[0]
        x = np.arange(codebook_size)
        ax1.bar(x, sorted_usage, color=self.colors['primary'], alpha=0.7, width=1.0)
        ax1.set_xlabel('Code Index (sorted)')
        ax1.set_ylabel('Usage Count')
        ax1.set_title(f'Codebook Usage (Active: {used_codes}/{codebook_size})')
        ax1.set_xlim([0, codebook_size])
        ax1.grid(True, alpha=0.3, axis='y')
        
        if dead_codes > 0:
            ax1.axvspan(used_codes, codebook_size, alpha=0.3, color=self.colors['danger'])
        
        # Cumulative distribution
        ax2 = axes[1]
        cumulative = np.cumsum(sorted_usage) / (total + 1e-10)
        ax2.plot(x / codebook_size * 100, cumulative * 100, 
                color=self.colors['primary'], linewidth=2)
        ax2.plot([0, 100], [0, 100], 'k--', linewidth=1, label='Uniform')
        ax2.fill_between(x / codebook_size * 100, cumulative * 100, 
                        alpha=0.3, color=self.colors['primary'])
        ax2.set_xlabel('% of Codes')
        ax2.set_ylabel('% of Tokens')
        ax2.set_title(f'Cumulative Usage (Perplexity: {perplexity:.1f})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0, 100])
        ax2.set_ylim([0, 100])
        
        fig.suptitle(f'Codebook Health (Step {step})', fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        img = fig_to_image(fig)
        self.writer.add_image(f"images/{tag}_histogram", img, step, dataformats='HWC')
        plt.close(fig)
        
        # Also log the usage as a histogram directly
        self.writer.add_histogram(f"{tag}/usage_distribution", usage, step)
    
    # =========================================================================
    # Embedding Visualization (t-SNE / PCA)
    # =========================================================================
    
    def log_embedding_tsne(
        self,
        embeddings: torch.Tensor,
        usage: Optional[torch.Tensor] = None,
        step: int = 0,
        method: str = 'auto',
        perplexity: int = 30,
        tag: str = "embeddings"
    ):
        """
        Log t-SNE or PCA visualization of codebook embeddings.
        
        Args:
            embeddings: Codebook vectors [K, D]
            usage: Usage count for each code [K] (for coloring)
            step: Global step
            method: 'tsne', 'pca', or 'auto' (auto selects based on K)
            perplexity: t-SNE perplexity
            tag: Tag for the image
        """
        if not self.enabled or not HAS_SKLEARN:
            if not HAS_SKLEARN:
                print("[TensorBoard] Warning: sklearn not available for t-SNE/PCA")
            return
        
        # Convert to numpy
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().numpy()
        if usage is not None and isinstance(usage, torch.Tensor):
            usage = usage.detach().cpu().numpy()
        
        K, D = embeddings.shape
        
        # Auto-select method
        if method == 'auto':
            method = 'tsne' if K > perplexity * 3 and K <= 5000 else 'pca'
        
        # Dimensionality reduction
        if method == 'tsne' and K > perplexity * 3:
            try:
                reducer = TSNE(n_components=2, perplexity=perplexity, 
                              random_state=42)
                reduced = reducer.fit_transform(embeddings)
            except Exception as e:
                print(f"[TensorBoard] t-SNE failed, falling back to PCA: {e}")
                method = 'pca'
                reducer = PCA(n_components=2)
                reduced = reducer.fit_transform(embeddings)
        else:
            method = 'pca'
            reducer = PCA(n_components=2)
            reduced = reducer.fit_transform(embeddings)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Usage-colored scatter
        ax1 = axes[0]
        if usage is not None:
            log_usage = np.log1p(usage)
            scatter = ax1.scatter(reduced[:, 0], reduced[:, 1], 
                                 c=log_usage, cmap='viridis',
                                 s=20, alpha=0.7)
            plt.colorbar(scatter, ax=ax1, label='Log(1 + usage)')
        else:
            ax1.scatter(reduced[:, 0], reduced[:, 1], 
                       color=self.colors['primary'], s=20, alpha=0.7)
        ax1.set_xlabel(f'{method.upper()} 1')
        ax1.set_ylabel(f'{method.upper()} 2')
        ax1.set_title(f'Embeddings ({method.upper()}) - Usage Colored')
        ax1.grid(True, alpha=0.3)
        
        # Active vs dead codes
        ax2 = axes[1]
        if usage is not None:
            active_mask = usage > 0
            dead_mask = usage == 0
            
            if active_mask.any():
                ax2.scatter(reduced[active_mask, 0], reduced[active_mask, 1],
                           color=self.colors['success'], s=20, alpha=0.7, label='Active')
            if dead_mask.any():
                ax2.scatter(reduced[dead_mask, 0], reduced[dead_mask, 1],
                           color=self.colors['danger'], s=20, alpha=0.7, label='Dead')
            ax2.legend()
        else:
            ax2.scatter(reduced[:, 0], reduced[:, 1],
                       color=self.colors['primary'], s=20, alpha=0.7)
        ax2.set_xlabel(f'{method.upper()} 1')
        ax2.set_ylabel(f'{method.upper()} 2')
        ax2.set_title('Active vs Dead Codes')
        ax2.grid(True, alpha=0.3)
        
        fig.suptitle(f'Codebook Embeddings (K={K}, D={D}, Step {step})', 
                    fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        img = fig_to_image(fig)
        self.writer.add_image(f"images/{tag}_tsne", img, step, dataformats='HWC')
        plt.close(fig)
    
    def log_latent_distribution(
        self,
        latents: torch.Tensor,
        step: int,
        tag: str = "latents"
    ):
        """
        Log distribution of latent representations.
        
        Args:
            latents: Latent vectors [B, D] or [B, N, D]
            step: Global step
            tag: Tag for the histogram
        """
        if not self.enabled:
            return
        
        if isinstance(latents, torch.Tensor):
            latents = latents.detach().cpu().numpy()
        
        latents_flat = latents.flatten()
        
        # Log histogram
        self.writer.add_histogram(f"{tag}/distribution", latents_flat, step)
        
        # Log stats
        self.writer.add_scalar(f"{tag}/mean", np.mean(latents_flat), step)
        self.writer.add_scalar(f"{tag}/std", np.std(latents_flat), step)
        self.writer.add_scalar(f"{tag}/min", np.min(latents_flat), step)
        self.writer.add_scalar(f"{tag}/max", np.max(latents_flat), step)
    
    # =========================================================================
    # Loss Breakdown Visualization
    # =========================================================================
    
    def log_loss_breakdown(
        self,
        loss_dict: Dict[str, float],
        step: int,
        prefix: str = "loss"
    ):
        """
        Log loss breakdown as stacked scalars.
        
        Args:
            loss_dict: Dict of loss name -> value
            step: Global step
            prefix: Tag prefix
        """
        if not self.enabled:
            return
        
        for name, value in loss_dict.items():
            if value is not None and not np.isnan(value):
                self.writer.add_scalar(f"{prefix}/{name}", value, step)
    
    def log_loss_pie_chart(
        self,
        loss_dict: Dict[str, float],
        step: int,
        tag: str = "loss_breakdown"
    ):
        """
        Log loss breakdown as a pie chart.
        
        Args:
            loss_dict: Dict of loss component name -> value
            step: Global step
            tag: Tag for the image
        """
        if not self.enabled or not loss_dict:
            return
        
        # Filter positive losses
        losses = {k: v for k, v in loss_dict.items() 
                  if v is not None and v > 0 and not np.isnan(v)}
        
        if not losses:
            return
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        labels = [k.replace('_loss', '').replace('_', ' ').title() for k in losses.keys()]
        values = list(losses.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(values)))
        
        wedges, texts, autotexts = ax.pie(
            values, labels=labels, autopct='%1.1f%%',
            colors=colors, startangle=90
        )
        ax.set_title(f'Loss Components (Step {step})')
        
        plt.tight_layout()
        
        img = fig_to_image(fig)
        self.writer.add_image(f"images/{tag}", img, step, dataformats='HWC')
        plt.close(fig)
    
    # =========================================================================
    # Training Progress Summary
    # =========================================================================
    
    def log_text_summary(
        self,
        text: str,
        step: int,
        tag: str = "summary"
    ):
        """
        Log text summary.
        
        Args:
            text: Summary text
            step: Global step
            tag: Tag for the text
        """
        if not self.enabled:
            return
        
        self.writer.add_text(tag, text, step)
    
    def log_hparams(
        self,
        hparams: Dict[str, Any],
        metrics: Dict[str, float]
    ):
        """
        Log hyperparameters with final metrics.
        
        Args:
            hparams: Dict of hyperparameter name -> value
            metrics: Dict of final metric name -> value
        """
        if not self.enabled:
            return
        
        # Convert to compatible types
        clean_hparams = {}
        for k, v in hparams.items():
            if isinstance(v, (bool, int, float, str)):
                clean_hparams[k] = v
            else:
                clean_hparams[k] = str(v)
        
        clean_metrics = {k: float(v) for k, v in metrics.items() 
                        if isinstance(v, (int, float))}
        
        self.writer.add_hparams(clean_hparams, clean_metrics)
    
    def flush(self):
        """Flush the TensorBoard writer."""
        if self.writer:
            self.writer.flush()
