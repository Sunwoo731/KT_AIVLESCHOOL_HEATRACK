import numpy as np
import rasterio
from shapely.geometry import Point
import logging

class Preprocessor:
    def __init__(self, config):
        self.patch_size = config['models']['autoencoder']['patch_size']
    
    def normalize_image(self, data):
        """Normalize data to [0, 1]."""
        d_min, d_max = np.nanmin(data), np.nanmax(data)
        if d_max == d_min: return np.zeros_like(data)
        data_norm = (data - d_min) / (d_max - d_min + 1e-6)
        return np.nan_to_num(data_norm, nan=0.0)

    def extract_patches(self, img_data, n_samples=3000):
        """Extract random patches from image."""
        rows, cols = img_data.shape
        patches = []
        
        rand_r = np.random.randint(0, rows - self.patch_size, n_samples)
        rand_c = np.random.randint(0, cols - self.patch_size, n_samples)
        
        for r, c in zip(rand_r, rand_c):
            patch = img_data[r : r+self.patch_size, c : c+self.patch_size]
            if patch.shape == (self.patch_size, self.patch_size):
                patches.append(patch.flatten())
                
        return np.array(patches)

    def inject_synthetic_leaks(self, normal_patches, n_leaks=200):
        """Inject heat into normal patches to simulate leaks."""
        indices = np.random.choice(len(normal_patches), n_leaks, replace=False)
        base_patches = normal_patches[indices].copy().reshape(-1, self.patch_size, self.patch_size)
        
        leak_patches = []
        center = self.patch_size // 2
        
        for patch in base_patches:
            heat_intensity = np.random.uniform(0.3, 0.6)
            patch[center, center] += heat_intensity
            patch = np.clip(patch, 0.0, 1.0)
            leak_patches.append(patch.flatten())
            
        return np.array(leak_patches)
