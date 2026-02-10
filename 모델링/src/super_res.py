
import numpy as np
import cv2
from sklearn.linear_model import LinearRegression

class SuperResolutionModule:
    def __init__(self, target_resolution=10):
        self.target_res = target_resolution

    def bicubic_upsample(self, image_np, scale_factor, target_shape=None):
        """
        Upsamples an image using Bicubic Interpolation.
        Input: (C, H, W) or (H, W, C) numpy array
        target_shape: (H, W) tuple for precise matching
        Output: Upsampled numpy array
        """
        # Ensure format is (H, W, C) for cv2
        transpose_needed = False
        if image_np.shape[0] < image_np.shape[2]: # likely (C, H, W)
             image_np = np.transpose(image_np, (1, 2, 0))
             transpose_needed = True
        
        h, w = image_np.shape[:2]
        
        if target_shape:
            new_h, new_w = target_shape
        else:
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        
        upsampled = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # If input had channels and cv2.resize dropped the 1-channel dim
        if len(image_np.shape) == 3 and len(upsampled.shape) == 2:
            upsampled = upsampled[:, :, np.newaxis]
            
        if transpose_needed:
            return np.transpose(upsampled, (2, 0, 1))
        
        return upsampled

    def swinir_upsample(self, thermal_lr, s2_hr, weights_path=None):
        """
        Intelligent Super-Resolution using SwinIR.
        thermal_lr: (1, H_lr, W_lr) or (H_lr, W_lr)
        s2_hr: (C_s2, H_hr, W_hr) - Guidance Map
        """
        import torch
        from src.lst_swinir import SwinIR_LST
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Instantiate Model (7 channels: Thermal + S2)
        model = SwinIR_LST(in_chans=7).to(device)
        
        # 2. Load Weights if specified
        if weights_path and os.path.exists(weights_path):
            try:
                state_dict = torch.load(weights_path, map_location=device)
                model.load_state_dict(state_dict)
                # print(f"Loaded SwinIR weights from {weights_path}")
            except Exception as e:
                print(f"Warning: Failed to load SwinIR weights: {e}")
        
        model.eval()

        # 3. Preprocess Inputs
        # thermal_lr needs to be upsampled to match s2_hr first (as SwinIR-LST expects aligned inputs)
        # or we could implement a native upsampler in SwinIR if trained that way.
        # Based on lst_swinir.py, it's a "restoration" (upscale=1) wrapper.
        
        target_h, target_w = s2_hr.shape[1], s2_hr.shape[2]
        thermal_up = self.bicubic_upsample(thermal_lr, scale_factor=None, target_shape=(target_h, target_w))
        
        if len(thermal_up.shape) == 2:
            thermal_up = thermal_up[np.newaxis, :, :]
            
        # Stack: Guidance (6) + Thermal (1)
        input_stack = np.concatenate([s2_hr, thermal_up], axis=0) # (7, H, W)
        input_tensor = torch.from_numpy(input_stack).float().unsqueeze(0).to(device)
        
        # 4. Inference (with padding for window size)
        window_size = 8 # Default in SwinIR_LST
        _, _, h, w = input_tensor.shape
        mod_pad_h = (window_size - h % window_size) % window_size
        mod_pad_w = (window_size - w % window_size) % window_size
        input_padded = torch.nn.functional.pad(input_tensor, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        
        with torch.no_grad():
            output_padded = model(input_padded) # (1, 1, H+, W+)
            
        # Crop back
        output = output_padded[:, :, :h, :w]
        
        return output.squeeze(0).cpu().numpy() # (1, H, W)

    def generate_confidence_map(self, image_shape, source_type):
        """
        Generates a confidence map based on source satellite.
        S1, S2 -> High Confidence (1.0)
        Landsat, S3 -> Lower Confidence (0.5)
        """
        if len(image_shape) == 3:
            h, w = image_shape[1], image_shape[2] # (C, H, W) format expected for tensors
        else:
            h, w = image_shape
            
        if source_type in ['S1', 'S2']:
            weight = 1.0
        else:
            weight = 0.5
        
        return np.full((h, w), weight, dtype=np.float16)

class GuidedUpsampler(SuperResolutionModule):
    def __init__(self):
        super().__init__()
        
    def compute_ndvi(self, s2_image):
        """
        Compute NDVI from Sentinel-2 image (H, W, C).
        Assuming bands are ordered such that Red is index 2, NIR is index 3 (B2, B3, B4, B8 ordering from data_collector)
        Wait, data_collector says bands=['B2', 'B3', 'B4', 'B8'].
        B4 = Red (idx 2), B8 = NIR (idx 3).
        """
        # (C, H, W) or (H, W, C)?
        # Let's handle both or assume (H,W,C) as per typical cv2 usage, but data_collector reads as (C,H,W) with rasterio usually?
        # Rasterio reads (C, H, W).
        
        red = s2_image[2, :, :]
        nir = s2_image[3, :, :]
        
        ndvi = (nir - red) / (nir + red + 1e-10)
        return ndvi

    def sharpen_thermal(self, thermal_low_res, s2_high_res):
        """
        TsHARP-like sharpening using NDVI.
        thermal_low_res: (1, H_lr, W_lr)
        s2_high_res: (4, H_hr, W_hr)
        """
        # 1. Compute High-Res NDVI (10m)
        ndvi_high = self.compute_ndvi(s2_high_res) #(H_hr, W_hr)
        
        # 2. Upsample Thermal to High Res (Bicubic) for residual calc later
        target_shape = ndvi_high.shape
        thermal_upsampled = self.bicubic_upsample(thermal_low_res, scale_factor=None, target_shape=target_shape)
        
        # 3. Downsample High-Res NDVI to match Low-Res Thermal spatial resolution
        # To fit regression model
        h_lr, w_lr = thermal_low_res.shape[1], thermal_low_res.shape[2]
        ndvi_low = cv2.resize(ndvi_high, (w_lr, h_lr), interpolation=cv2.INTER_AREA)
        
        # 4. Fit Linear Regression: Thermal_LR ~ NDVI_LR
        # Flatten
        X = ndvi_low.reshape(-1, 1)
        y = thermal_low_res.reshape(-1)
        
        reg = LinearRegression()
        reg.fit(X, y)
        
        # 5. Predict High-Res Thermal using High-Res NDVI
        X_high = ndvi_high.reshape(-1, 1)
        thermal_high_pred = reg.predict(X_high).reshape(target_shape)
        
        # 6. Residual Correction
        # Residual at Low Res = Actual_LR - Predicted_LR_from_NDVI_Low
        thermal_low_pred = reg.predict(X).reshape(h_lr, w_lr)
        residual_low = thermal_low_res[0] - thermal_low_pred
        
        # Upsample Residual to High Res
        residual_high = cv2.resize(residual_low, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_CUBIC)
        
        # Final Sharpened Thermal = Predicted_High + Residual_High
        thermal_sharpened = thermal_high_pred + residual_high
        
        return thermal_sharpened[np.newaxis, :, :] # Return (1, H, W)
