from src.swinir_model import SwinIR
import torch.nn as nn

class SwinIR_LST(nn.Module):
    def __init__(self, upscale=1, in_chans=7, img_size=64, window_size=8, img_range=1.0):
        """
        Wrapper for SwinIR for Land Surface Temperature Super-Resolution.
        
        Args:
            upscale (int): Upscale factor. Default 1 (we assume inputs are already spatially aligned/interpolated or we use it for restoration).
                           If we want actual upsampling from LR, we should set this to 2, 4 etc.
                           However, for fusion (S2 is 10m, Landsat is 100m -> 10m), we typically concatenate Upsampled-Landsat + S2.
                           So the model sees 10m grids. Thus upscale=1 (Image Restoration mode).
            in_chans (int): Thermal (1) + S2 (6) = 7.
            img_size (int): Input patch size during training.
        """
        super(SwinIR_LST, self).__init__()
        
        # Configuration for "lightweight" or "balanced" restoration
        self.model = SwinIR(
            img_size=img_size,
            patch_size=1,
            in_chans=in_chans,
            embed_dim=60, # Reduced from 96 for efficiency
            depths=[6, 6, 6, 6],
            num_heads=[6, 6, 6, 6],
            window_size=window_size,
            mlp_ratio=2,
            upscale=upscale,
            img_range=img_range,
            upsampler='', # No upsampler at end, just restoration
            resi_connection='1conv'
        )
        
        # Output is in_chans by default in SwinIR if not specified? 
        # Actually in SwinIR __init__:
        # num_out_ch = in_chans (Line 655 provided file)
        # self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1) (Line 762 provided file)
        
        # We want output to be 1 channel (LST only), generally SwinIR restores inputs.
        # So we need to modify the last layer or use a wrapper.
        # Since I downloaded the file, I can't easily change the class def without editing it.
        # But wait, looking at `swinir_model.py`:
        # It defines `num_out_ch = in_chans` (Line 655).
        
        # We should override the last layer after instantiation.
        self.model.conv_last = nn.Conv2d(self.model.embed_dim, 1, 3, 1, 1)
        
    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    import torch
    # Test
    model = SwinIR_LST(in_chans=7)
    x = torch.randn(1, 7, 64, 64)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
