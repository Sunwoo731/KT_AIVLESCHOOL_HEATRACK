import sys
import os
import torch
import numpy as np
import traceback

# Add local model directory to path to import PrithviMAE
sys.path.append(os.path.join(os.getcwd(), "Prithvi_Model"))

# Try to import the class directly
try:
    from prithvi_mae import PrithviMAE
except ImportError as e:
    print(f"PrithviMAE import failed: {e}")
    PrithviMAE = None
except Exception as e:
    print(f"Unexpected error during import: {e}")
    # Print traceback for import error
    traceback.print_exc()
    PrithviMAE = None

from prithvi_utils import PrithviPreprocessor

def run_inference_local(data_dir, output_dir):
    if PrithviMAE is None:
        print("Cannot run: Model class not imported.")
        return

    print("Instantiating PrithviMAE locally...")
    try:
        # Default configuration values for 100M version
        config = {
            "img_size": 224,
            "patch_size": 16,
            "num_frames": 3,
            "in_chans": 6,
            "embed_dim": 768,
            "depth": 12,
            "num_heads": 12,
            "decoder_embed_dim": 512,
            "decoder_depth": 8,
            "decoder_num_heads": 16,
            "mlp_ratio": 4,
            "norm_pix_loss": False,
            "coords_encoding": [],
            "coords_scale_learn": False
        }

        # Instantiate
        model = PrithviMAE(**config)
        print("Model instantiated successfully.")
        
        # Validate and Load Weights
        weights_path = os.path.join("Prithvi_Model", "Prithvi_EO_V1_100M.pt")
        if not os.path.exists(weights_path):
             weights_path = os.path.join("Prithvi_Model", "Prithvi_100M.pt")
        
        if os.path.exists(weights_path):
            print(f"Loading weights from {weights_path}...")
            state_dict = torch.load(weights_path, map_location='cpu')
            
            # Key cleaning for compatibility
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k.replace("module.", "")
                # If weights are from a checkpoint containing 'model.' prefix
                name = name.replace("model.", "")
                new_state_dict[name] = v
                
            msg = model.load_state_dict(new_state_dict, strict=False)
            print(f"Weights loaded. Missing keys: {len(msg.missing_keys)}, Unexpected keys: {len(msg.unexpected_keys)}")
        else:
            print("WARNING: No weights file found! Using random weights.")

        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print(f"Model on {device}")

    except Exception:
        print("Error during model instantiation or weight loading:")
        traceback.print_exc()
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    processor = PrithviPreprocessor(data_dir)
    print(f"Processing {len(processor.pairs)} files...")
    
    from tqdm import tqdm
    for i in tqdm(range(len(processor.pairs))):
        # Remove limit
        # if count >= 3: break
        
        result = processor.load_multispectral_input(i)
        if result is None: continue
        
        input_tensor, fname = result
        
        # S2 (6 bands) -> Prithvi
        prithvi_input = input_tensor[:6, :, :].unsqueeze(0).unsqueeze(2) # (1, 6, 1, H, W)
        prithvi_input = prithvi_input.to(device)
        
        with torch.no_grad():
            try:
                # Get LAST LAYER features
                features = model.forward_features(prithvi_input)
                
                # Reshape to spatial grid manually
                # features has shape (B, Tokens, Dim)
                # Tokens = 1 (CLS) + H_grid * W_grid * T_grid
                last_feat = features[-1]
                x_no_token = last_feat[:, 1:, :] # Remove CLS token
                
                # Calculate grid dimensions based on input patch count
                # patch_size=16. Input shape (B, C, T, H, W)
                grid_h = prithvi_input.shape[-2] // 16
                grid_w = prithvi_input.shape[-1] // 16
                
                # Reshape to (H_grid, W_grid, Dim)
                last_feat_spatial = x_no_token.reshape(grid_h, grid_w, -1)
                
                base_name = os.path.splitext(os.path.basename(fname))[0]
                save_path = os.path.join(output_dir, f"{base_name}_features.npy")
                np.save(save_path, last_feat_spatial.cpu().numpy())
                # print(f"Saved {base_name} as {last_feat_spatial.shape}")
                
            except Exception as e:
                print(f"Inference failed for {fname}: {e}")
                traceback.print_exc()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to raw data directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output features directory")
    args = parser.parse_args()
    
    run_inference_local(args.data_dir, args.output_dir)
