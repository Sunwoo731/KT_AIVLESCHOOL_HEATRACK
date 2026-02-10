import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import rasterio
from src.lst_swinir import SwinIR_LST
from src.prithvi_utils import PrithviPreprocessor
# from src.losses import CharbonnierLoss, PerceptualLoss # To be implemented

class LSTDataset(Dataset):
    def __init__(self, data_dir, patch_size=64):
        self.processor = PrithviPreprocessor(data_dir)
        self.patch_size = patch_size
        
    def __len__(self):
        return len(self.processor.pairs)
        
    def __getitem__(self, idx):
        # Load (C, H, W) tensor: S2(6) + L8(1 Thermal)
        # We need to return:
        # LR (Input): S2 (6 bands) + Upsampled L8 (1 band) -> 7 channels
        # HR (Target): The original L8 Thermal (but wait, we don't have HR thermal ground truth)
        # Actually, the task is: 100m Thermal -> 10m Thermal.
        # Self-supervised strategy:
        # Downsample 100m Thermal to 1km (?) or just train on reconstruction?
        # Or: Use "Input" as S2(10m) + Bicubic-L8(10m) and "Target" as ... we don't have 10m Thermal ground truth.
        
        # Alternative Strategy (k-folding / spatial consistency):
        # 1. We assume the network learns to transfer texture from S2 to Thermal.
        # 2. Training objective: Reconstruct the *original 100m* thermal when downsampled.
        #    Loss = | Downsample(Predicted_10m) - Original_100m | + | Gradient(Predicted) - Gradient(S2) | (Texture transfer)
        
        # For this implementation, let's setup the mechanism to load the data first.
        # We'll stick to a simple Reconstruction Loss on the *original scale* for now, 
        # or if we had pairs (which we don't), we'd use them.
        
        result = self.processor.load_multispectral_input(idx)
        if result is None:
            # Handle error safely
            return torch.zeros(7, self.patch_size, self.patch_size), torch.zeros(1, self.patch_size, self.patch_size)
            
        input_tensor, fname = result
        # input_tensor is (7, H, W). Channel 6 is Thermal.
        
        # Random Crop
        h, w = input_tensor.shape[1], input_tensor.shape[2]
        if h > self.patch_size and w > self.patch_size:
            x = np.random.randint(0, h - self.patch_size)
            y = np.random.randint(0, w - self.patch_size)
            crop = input_tensor[:, x:x+self.patch_size, y:y+self.patch_size]
        else:
            crop = input_tensor[:, :self.patch_size, :self.patch_size]
            # Pad if needed
            # ...
            
        # For training, let's just return the crop.
        # Input: All 7 channels (assuming L8 is already upsampled to 10m grid by PrithviPreprocessor via resize)
        # Target: The Thermal channel itself (as a self-reconstruction identity task? No.)
        
        # Paper approach: "Guided Super-Resolution"
        # Input: Guidance (S2) + LowRes (L8)
        # Output: HighRes (L8)
        # Since no Ground Truth, we can downgrade L8 further (100m -> 1km) and train to recover 100m.
        # But that might not map to 100m->10m well.
        
        # Let's assume for now we return the crop and handle loss logic in the loop.
        return crop

def train(data_dir, epochs=10, batch_size=4, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = LSTDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = SwinIR_LST(in_chans=7).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss() # Charbonnier is better but L1 is close
    
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for i, batch in enumerate(dataloader):
            batch = batch.to(device) # (B, 7, H, W)
            
            # Input: S2 features + Thermal
            # We want to refine the Thermal channel (Index 6)
            
            # Synthetically downsample the Thermal channel to create a training pair?
            # Or just train as an Autoencoder for now to learn the manifold?
            
            # Current simplified objective: Denoising/Refining.
            # Target = Original Batch (Identity) ? No that leads to identity mapping.
            
            # Let's implement the "Downsample Loss" approach:
            # Pred_HR = Model(Input)
            # Loss = L1(Downsample(Pred_HR), Original_LR) + TV(Pred_HR)
            
            inputs = batch
            optimizer.zero_grad()
            
            outputs = model(inputs) # (B, 1, H, W)
            
            # Simple consistency loss for setup check (Input Thermal ~ Output Thermal)
            # This is NOT SR yet, but checks the pipeline.
            target_thermal = batch[:, 6:7, :, :]
            
            loss = criterion(outputs, target_thermal)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i}/{len(dataloader)}], Loss: {loss.item():.4f}")
                
        print(f"Epoch {epoch+1} Average Loss: {total_loss / len(dataloader):.4f}")
        
    # Save model
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/swinir_lst_latest.pth")
    print("Training finished. Model saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()
    
    train(args.data_dir)
