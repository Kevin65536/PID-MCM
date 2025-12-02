import torch
import numpy as np
from torch.utils.data import Dataset

class PIDSyntheticDataset(Dataset):
    """
    Synthetic dataset for verifying PID-MCM (ELP) framework.
    
    Generates data from latent sources:
    - w_r: Redundant information (shared by X1, X2)
    - w_u1: Unique information for X1
    - w_u2: Unique information for X2
    - w_s: Synergistic information (interaction required)
    
    X1 = f1(w_r, w_u1, w_s)
    X2 = f2(w_r, w_u2, w_s)
    """
    def __init__(self, num_samples=10000, dim_latent=16, dim_obs=64, noise_level=0.1, seed=42):
        super().__init__()
        self.num_samples = num_samples
        self.dim_latent = dim_latent
        self.dim_obs = dim_obs
        self.noise_level = noise_level
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        self._generate_data()
        
    def _generate_data(self):
        # 1. Generate independent latent sources
        # Shape: [N, D_latent]
        self.w_r = torch.randn(self.num_samples, self.dim_latent)
        self.w_u1 = torch.randn(self.num_samples, self.dim_latent)
        self.w_u2 = torch.randn(self.num_samples, self.dim_latent)
        self.w_s = torch.randn(self.num_samples, self.dim_latent)
        
        # 2. Define mixing matrices (Linear transformation for simplicity, can be non-linear)
        # We project [w_r, w_u, w_s] -> X
        
        # Projection for X1: Uses R, U1, S
        self.P1_r = torch.randn(self.dim_latent, self.dim_obs)
        self.P1_u = torch.randn(self.dim_latent, self.dim_obs)
        self.P1_s = torch.randn(self.dim_latent, self.dim_obs)
        
        # Projection for X2: Uses R, U2, S
        self.P2_r = torch.randn(self.dim_latent, self.dim_obs) # Different projection for R? 
        # Ideally R implies shared content. If X1 and X2 are different modalities, 
        # the *content* is same but *manifestation* (projection) might differ.
        # But to make it strictly redundant, there should be a mapping f1^-1(X1) approx f2^-1(X2).
        
        self.P2_u = torch.randn(self.dim_latent, self.dim_obs)
        self.P2_s = torch.randn(self.dim_latent, self.dim_obs)
        
        # 3. Generate Observations
        # X1 = w_r @ P1_r + w_u1 @ P1_u + w_s @ P1_s + noise
        self.x1 = (self.w_r @ self.P1_r + 
                   self.w_u1 @ self.P1_u + 
                   self.w_s @ self.P1_s + 
                   torch.randn(self.num_samples, self.dim_obs) * self.noise_level)
                   
        # X2 = w_r @ P2_r + w_u2 @ P2_u + w_s @ P2_s + noise
        self.x2 = (self.w_r @ self.P2_r + 
                   self.w_u2 @ self.P2_u + 
                   self.w_s @ self.P2_s + 
                   torch.randn(self.num_samples, self.dim_obs) * self.noise_level)
        
        # 4. Generate Task Labels (Optional, for downstream verification)
        # Task 1: Relies on Redundancy (R) -> y = f(w_r)
        self.y_red = (self.w_r.sum(dim=1) > 0).float()
        
        # Task 2: Relies on Synergy (S) -> y = XOR(w_u1, w_u2) or just f(w_s)
        # Here we define synergy as information accessible only if we have w_s.
        # Since w_s is mixed into X1 and X2, we need to ensure it's not recoverable from just one.
        # But in this linear mixing, w_s IS recoverable from X1 alone if dim_obs >= 3*dim_latent.
        # To make it "Synergistic" in the PID sense (X1, X2 -> Y), Y must depend on interactions.
        # But for ELP representation learning, we just want to recover the latents.
        
        # Let's stick to recovering latents for now.
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return {
            'x1': self.x1[idx],
            'x2': self.x2[idx],
            'latents': {
                'r': self.w_r[idx],
                'u1': self.w_u1[idx],
                'u2': self.w_u2[idx],
                's': self.w_s[idx]
            }
        }

if __name__ == "__main__":
    # Test
    dataset = PIDSyntheticDataset()
    sample = dataset[0]
    print("X1 shape:", sample['x1'].shape)
    print("X2 shape:", sample['x2'].shape)
    print("Latent R shape:", sample['latents']['r'].shape)
