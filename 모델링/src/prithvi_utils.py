import os
import re
import datetime
import torch
import numpy as np
import rasterio
# Check if skimage is available, if not use scipy or naive resizing
try:
    from skimage.transform import resize
except ImportError:
    resize = None

class PrithviPreprocessor:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.s2_files = sorted([f for f in os.listdir(data_dir) if f.startswith('S2') and f.endswith('.tif')])
        self.l8_files = sorted([f for f in os.listdir(data_dir) if f.startswith('Landsat') and f.endswith('.tif')])
        self.pairs = self._match_files_by_date()

    def _parse_date(self, filename):
        """
        Parses date from filename. 
        Expected formats:
        - Sentinel-2: ...YYYYMMDD... (e.g. S2_20190103.tif or S2A_MSIL2A_20191109...)
        - Landsat: ...YYYYDDD... (Julian day, e.g. Landsat_2019013.tif or Landsat_2019123.tif)
        """
        try:
            # 1. Try to find 8-digit date (YYYYMMDD)
            match8 = re.search(r'(\d{8})', filename)
            if match8:
                date_str = match8.group(1)
                # Verify it's a valid Gregorian date
                try:
                    return datetime.datetime.strptime(date_str, "%Y%m%d")
                except ValueError:
                    pass # Not a valid YYYYMMDD, try Julian
            
            # 2. Try to find 7-digit date (YYYYDDD) - often in Landsat
            match7 = re.search(r'(\d{7})', filename)
            if match7:
                date_str = match7.group(1)
                try:
                    return datetime.datetime.strptime(date_str, "%Y%j")
                except ValueError:
                    pass

            # 3. Fallback for Sentinel-2 specific naming patterns if needed
            # But usually \d{8} covers it.

        except Exception as e:
            print(f"Skipping file {filename}: {e}")
            
        return None

    def _match_files_by_date(self, max_delta_days=7):
        pairs = []
        l8_dates = {self._parse_date(f): f for f in self.l8_files if self._parse_date(f)}
        sorted_l8_dates = sorted(l8_dates.keys())
        
        # Debug: Print sample of parsed dates
        if sorted_l8_dates:
             print(f"Sample Landsat dates: {[d.strftime('%Y-%m-%d') for d in sorted_l8_dates[:3]]}")

        for s2_f in self.s2_files:
            s2_date = self._parse_date(s2_f)
            if not s2_date: continue
            
            best_match = None
            min_diff = float('inf')
            
            for l8_d in sorted_l8_dates:
                diff = abs((s2_date - l8_d).days)
                if diff <= max_delta_days and diff < min_diff:
                    min_diff = diff
                    best_match = l8_dates[l8_d]
            
            if best_match:
                pairs.append((s2_f, best_match))
        
        print(f"Matched {len(pairs)} S2-Landsat pairs (max delta: {max_delta_days} days).")
        return pairs

    def load_multispectral_input(self, idx):
        if idx >= len(self.pairs):
            return None
            
        s2_filename, l8_filename = self.pairs[idx]
        s2_path = os.path.join(self.data_dir, s2_filename)
        l8_path = os.path.join(self.data_dir, l8_filename)
        
        try:
            with rasterio.open(s2_path) as src:
                s2_data = src.read() # (C, H, W)
                profile = src.profile

            with rasterio.open(l8_path) as src:
                l8_data = src.read() # (C, H, W)
                # print(f"DEBUG: L8 Shape for {l8_filename}: {l8_data.shape}")
            
            # Resize L8 to match S2
            # L8 Shape: (C_l8, H_l8, W_l8). S2 Shape: (C_s2, H_s2, W_s2).
            if s2_data.shape[1:] != l8_data.shape[1:]:
                if resize:
                    # resize expects (H, W, C)
                    l8_trans = l8_data.transpose(1, 2, 0)
                    l8_resized = resize(l8_trans, (s2_data.shape[1], s2_data.shape[2]), anti_aliasing=True)
                    # Move Channel back: (C, H, W)
                    l8_ready = l8_resized.transpose(2, 0, 1)
                else:
                     print("Warning: scikit-image not found, cannot resize Landsat.")
                     return None
            else:
                l8_ready = l8_data

            # Stack: Channels = S2_Bands + L8_Thermal
            input_tensor = np.concatenate([s2_data, l8_ready], axis=0)
            return torch.from_numpy(input_tensor).float(), s2_filename
            
        except Exception as e:
            print(f"Error loading {s2_filename}: {e}")
            return None

if __name__ == "__main__":
    # Test with Jeongja-dong data
    base_dir = r"D:\빅프로젝트\모델링\20191215 정자동\data\raw"
    if os.path.exists(base_dir):
        processor = PrithviPreprocessor(base_dir)
        result = processor.load_multispectral_input(0)
        if result is not None:
            tensor, fname = result
            print(f"Successfully created Prithvi input cube from {fname}")
            print(f"Shape: {tensor.shape}")
        else:
            print("No matching file pair found or load failed.")
    else:
        print(f"Directory not found: {base_dir}")
