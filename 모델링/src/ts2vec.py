
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter

class TS2Vec(nn.Module):
    def __init__(self, input_dim, output_dim=128):
        super(TS2Vec, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Simple Encoder for now
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        """
        x: (Batch, Channels, Time) -> But here we might process per pixel
        Let's assume input is (N, C, T) or (N, T, C). 
        TS2Vec usually handles (B, T, C).
        """
        # (B, T, C)
        # For simplicity in this restoration:
        return self.encoder(x)

    def train_step(self, x, weights=None):
        """
        Mock training step.
        """
        # x shape?
        return 0.1 # Mock loss

    def encode(self, x):
        """
        Returns embeddings.
        """
        with torch.no_grad():
            # If x is (N, C, T), pool over time?
            # Or simplified: if x is (N, C), direct encode.
            # Assuming main.py passed (H*W, C, T) roughly.
            # Let's map to random embedding for restoration stub.
            return torch.randn(x.shape[0], self.output_dim)

def spatial_smoothing(embedding_map, sigma=0.5):
    """
    Applies Gaussian smoothing to the embedding map (H, W, Dim) or (H, W).
    """
    print(f"DEBUG SHAPE: {embedding_map.shape}")
    if len(embedding_map.shape) == 2:
        return gaussian_filter(embedding_map, sigma=sigma)
    
    h, w, c = embedding_map.shape
    smoothed = np.zeros_like(embedding_map)
    for i in range(c):
        smoothed[:, :, i] = gaussian_filter(embedding_map[:, :, i], sigma=sigma)
    return smoothed
