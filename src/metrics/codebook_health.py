"""
Codebook health metrics for tokenizer evaluation.

These metrics help diagnose codebook collapse and measure 
the effective utilization of the learned discrete representations.
"""

import torch
from typing import Dict, Optional


def compute_perplexity(indices: torch.Tensor, codebook_size: int) -> float:
    """
    Compute codebook perplexity (exponential of entropy).
    
    Perplexity measures how uniformly the codebook is being used.
    - Maximum = codebook_size (uniform usage)
    - Minimum = 1 (only one code used)
    
    Args:
        indices: Token indices [B, T'] or [N]
        codebook_size: Total number of codes in codebook
        
    Returns:
        Perplexity value (float)
    """
    flat = indices.flatten()
    usage = torch.bincount(flat, minlength=codebook_size).float()
    usage_prob = usage / (usage.sum() + 1e-10)
    
    # Entropy
    entropy = -(usage_prob * torch.log(usage_prob + 1e-10)).sum()
    
    # Perplexity = exp(entropy)
    perplexity = torch.exp(entropy)
    return perplexity.item()


def compute_code_utilization(indices: torch.Tensor, codebook_size: int) -> float:
    """
    Compute the ratio of active codes (codes that are used at least once).
    
    Args:
        indices: Token indices [B, T'] or [N]
        codebook_size: Total number of codes
        
    Returns:
        Utilization ratio in [0, 1]
    """
    flat = indices.flatten()
    usage = torch.bincount(flat, minlength=codebook_size)
    active_codes = (usage > 0).sum()
    return (active_codes / codebook_size).item()


def compute_dead_codes(indices: torch.Tensor, codebook_size: int) -> int:
    """
    Count codes that are never used.
    
    Args:
        indices: Token indices [B, T'] or [N]
        codebook_size: Total number of codes
        
    Returns:
        Number of dead codes
    """
    flat = indices.flatten()
    usage = torch.bincount(flat, minlength=codebook_size)
    return (usage == 0).sum().item()


def compute_usage_distribution(
    indices: torch.Tensor, 
    codebook_size: int,
    top_k: Optional[int] = None
) -> Dict[str, float]:
    """
    Compute detailed usage distribution statistics.
    
    Args:
        indices: Token indices [B, T'] or [N]
        codebook_size: Total number of codes
        top_k: If specified, compute top-k coverage
        
    Returns:
        Dict with distribution statistics
    """
    flat = indices.flatten()
    usage = torch.bincount(flat, minlength=codebook_size).float()
    usage_prob = usage / (usage.sum() + 1e-10)
    
    # Sort by usage
    sorted_usage, _ = usage_prob.sort(descending=True)
    
    result = {
        'max_usage': sorted_usage[0].item(),
        'min_nonzero_usage': sorted_usage[sorted_usage > 0][-1].item() if (sorted_usage > 0).any() else 0,
        'usage_std': usage_prob.std().item(),
        'gini_coefficient': _compute_gini(usage_prob).item(),
    }
    
    # Top-k coverage
    if top_k is not None:
        top_k = min(top_k, codebook_size)
        result[f'top_{top_k}_coverage'] = sorted_usage[:top_k].sum().item()
    
    return result


def _compute_gini(prob: torch.Tensor) -> torch.Tensor:
    """
    Compute Gini coefficient to measure inequality in usage distribution.
    - 0 = perfect equality (uniform usage)
    - 1 = perfect inequality (one code used for everything)
    """
    sorted_prob, _ = prob.sort()
    n = len(sorted_prob)
    
    if n == 0 or prob.sum() == 0:
        return torch.tensor(0.0)
    
    cumsum = torch.cumsum(sorted_prob, dim=0)
    return (2 * torch.arange(1, n + 1, device=prob.device).float() @ sorted_prob - (n + 1) * sorted_prob.sum()) / (n * sorted_prob.sum() + 1e-10)


def compute_codebook_health(
    indices: torch.Tensor, 
    codebook_size: int,
    include_distribution: bool = False,
    top_k: int = 10
) -> Dict[str, float]:
    """
    Compute comprehensive codebook health metrics.
    
    This is the main function for evaluating codebook quality.
    
    Args:
        indices: Token indices [B, T'] or [N]
        codebook_size: Total number of codes
        include_distribution: Whether to include detailed distribution stats
        top_k: Number of top codes for coverage computation
        
    Returns:
        Dict containing:
            - perplexity: Exponential of entropy
            - code_utilization: Ratio of active codes
            - dead_codes: Count of unused codes
            - active_codes: Count of used codes
            - (optional) distribution stats
    """
    flat = indices.flatten()
    usage = torch.bincount(flat, minlength=codebook_size).float()
    usage_prob = usage / (usage.sum() + 1e-10)
    
    # Perplexity
    entropy = -(usage_prob * torch.log(usage_prob + 1e-10)).sum()
    perplexity = torch.exp(entropy)
    
    # Utilization
    active_codes = (usage > 0).sum()
    utilization = active_codes / codebook_size
    
    # Dead codes
    dead_codes = (usage == 0).sum()
    
    result = {
        'perplexity': perplexity.item(),
        'code_utilization': utilization.item(),
        'dead_codes': dead_codes.item(),
        'active_codes': active_codes.item(),
    }
    
    if include_distribution:
        dist_stats = compute_usage_distribution(indices, codebook_size, top_k)
        result.update(dist_stats)
    
    return result


def check_health_thresholds(
    metrics: Dict[str, float],
    codebook_size: int,
    perplexity_ratio: float = 0.3,
    utilization_min: float = 0.2,
    dead_ratio_max: float = 0.3,
) -> Dict[str, bool]:
    """
    Check if codebook health metrics meet thresholds.
    
    Args:
        metrics: Output from compute_codebook_health
        codebook_size: Total codebook size
        perplexity_ratio: Minimum perplexity as ratio of codebook_size
        utilization_min: Minimum code utilization
        dead_ratio_max: Maximum ratio of dead codes
        
    Returns:
        Dict with pass/fail for each criterion
    """
    return {
        'perplexity_ok': metrics['perplexity'] >= perplexity_ratio * codebook_size,
        'utilization_ok': metrics['code_utilization'] >= utilization_min,
        'dead_codes_ok': metrics['dead_codes'] / codebook_size <= dead_ratio_max,
    }


if __name__ == "__main__":
    # Test codebook health metrics
    print("Testing codebook health metrics...")
    
    # Simulated indices (some collapse)
    codebook_size = 512
    indices = torch.randint(0, 100, (32, 64))  # Only using first 100 codes
    
    health = compute_codebook_health(indices, codebook_size, include_distribution=True)
    print(f"\nCodebook health:")
    for k, v in health.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    
    checks = check_health_thresholds(health, codebook_size)
    print(f"\nThreshold checks:")
    for k, v in checks.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    
    # Healthy case
    print("\n--- Healthy codebook simulation ---")
    healthy_indices = torch.randint(0, codebook_size, (32, 64))
    healthy_health = compute_codebook_health(healthy_indices, codebook_size)
    print(f"Perplexity: {healthy_health['perplexity']:.1f}")
    print(f"Utilization: {healthy_health['code_utilization']:.4f}")
