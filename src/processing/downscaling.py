import rasterio
from rasterio.enums import Resampling
import numpy as np
from scipy.ndimage import gaussian_filter
import logging
import os

class ThermalDownscaler:
    def __init__(self, config):
        self.config = config

    def downscale(self, lst_path, b4_path, b8_path, output_path):
        """
        Perform thermal downscaling using Sentinel-2 NDVI.
        LST (100m) -> 10m High Res
        """
        logging.info(f"Starting downscaling for {lst_path}")
        
        # 1. Load LST Reference
        with rasterio.open(lst_path) as src:
            lst_bounds = src.bounds
            lst_crs = src.crs
            lst_data = src.read(1)
            lst_transform = src.transform

        # 2. Define Target Grid (10m)
        width = int((lst_bounds.right - lst_bounds.left) / 10.0)
        height = int((lst_bounds.top - lst_bounds.bottom) / 10.0)
        
        target_transform = rasterio.transform.from_bounds(
            lst_bounds.left, lst_bounds.bottom, lst_bounds.right, lst_bounds.top, width, height
        )

        def reproject_to_target(path):
            with rasterio.open(path) as src:
                dest = np.zeros((height, width), dtype=np.float32)
                rasterio.warp.reproject(
                    source=rasterio.band(src, 1),
                    destination=dest,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=target_transform,
                    dst_crs=lst_crs,
                    resampling=Resampling.bilinear
                )
                return dest

        # 3. Process Bands
        logging.info("Reading Optical Bands...")
        b4 = reproject_to_target(b4_path)
        b8 = reproject_to_target(b8_path)
        
        # Upsample LST to target resolution (Baseline)
        lst_upsampled = np.zeros((height, width), dtype=np.float32)
        with rasterio.open(lst_path) as src:
            rasterio.warp.reproject(
                source=rasterio.band(src, 1),
                destination=lst_upsampled,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=target_transform,
                dst_crs=lst_crs,
                resampling=Resampling.bilinear
            )

        # 4. Calibration (NDVI based)
        logging.info("Calculating NDVI Sharpening...")
        ndvi = (b8 - b4) / (b8 + b4 + 1e-6)
        ndvi_low = gaussian_filter(ndvi, sigma=10)
        ndvi_detail = ndvi - ndvi_low
        
        # Coupling Factor (Empirical)
        coupling_factor = 5.0
        lst_high_res = lst_upsampled - (ndvi_detail * coupling_factor)

        # 5. Save
        kwargs = {
            'driver': 'GTiff',
            'crs': lst_crs,
            'transform': target_transform,
            'width': width,
            'height': height,
            'count': 1,
            'dtype': 'float32',
            'compress': 'lzw'
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with rasterio.open(output_path, 'w', **kwargs) as dst:
            dst.write(lst_high_res, 1)
            
        logging.info(f"Saved high-res thermal map to {output_path}")
        return lst_high_res
