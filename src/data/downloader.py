import ee
import geemap
import os
from datetime import datetime, timedelta
import logging

class SatelliteDownloader:
    def __init__(self, config):
        self.config = config
        self.output_dir = config['paths']['raw_satellite']
        self.project_id = config['satellite'].get('ee_project_id')
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        self._authenticate()

    def _authenticate(self):
        try:
            if self.project_id and self.project_id != "CHANGE_THIS_TO_YOUR_PROJECT_ID":
                ee.Initialize(project=self.project_id)
            else:
                ee.Initialize()
            logging.info("Google Earth Engine Initialized successfully.")
        except Exception as e:
            logging.error(f"Initialization failed: {e}")
            logging.info("Please run `ee.Authenticate()` manually if credential issues persist.")

    def download_for_location(self, name, lat, lon, date_str):
        logging.info(f"Processing: {name} ({date_str})")
        
        target_date = datetime.strptime(date_str, "%Y-%m-%d")
        start_date = (target_date - timedelta(days=21)).strftime("%Y-%m-%d")
        end_date = date_str
        
        roi = ee.Geometry.Point([lon, lat]).buffer(self.config['satellite']['roi_buffer_meters']).bounds()
        
        self._download_landsat(roi, start_date, end_date, name)
        self._download_sentinel(roi, start_date, end_date, name)

    def _download_landsat(self, roi, start_date, end_date, name):
        """Download Landsat 8/9 LST Data."""
        collection = self.config['satellite']['landsat_collection']
        l8 = ee.ImageCollection(collection) \
            .filterBounds(roi) \
            .filterDate(start_date, end_date) \
            .filter(ee.Filter.lt("CLOUD_COVER", 30)) \
            .sort("system:time_start", False) \
            .first()

        if l8 is None:
            logging.warning(f"[Landsat] No image found for {name}")
            return

        def to_celsius(img):
            # C02 L2 ST_B10 is DN. C * DN + Off - 273.15. 
            # Note: Standard is 0.00341802 * DN + 149.0 - 273.15
            return img.select('ST_B10').multiply(0.00341802).add(149.0).subtract(273.15).rename('LST')

        lst_img = to_celsius(l8)
        fname = f"{name.replace(' ', '_')}_LST.tif"
        out_path = os.path.join(self.output_dir, fname)
        
        # Check if already exists
        if os.path.exists(out_path):
             logging.info(f"[Landsat] Skipping (Exists): {fname}")
             return

        try:
            geemap.ee_export_image(lst_img, filename=out_path, scale=30, region=roi, file_per_band=False)
            logging.info(f"[Landsat] Saved: {fname}")
        except Exception as e:
            logging.error(f"[Landsat] Export failed: {e}")

    def _download_sentinel(self, roi, start_date, end_date, name):
        """Download Sentinel-2 Optical Data (B4, B8)."""
        collection = self.config['satellite']['sentinel_collection']
        s2 = ee.ImageCollection(collection) \
            .filterBounds(roi) \
            .filterDate(start_date, end_date) \
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20)) \
            .sort("system:time_start", False) \
            .first()

        if s2 is None:
            logging.warning(f"[Sentinel-2] No image found for {name}")
            return

        s2_bands = s2.select(['B4', 'B8'])
        fname = f"{name.replace(' ', '_')}_S2.tif"
        out_path = os.path.join(self.output_dir, fname)
        
        if os.path.exists(out_path):
             logging.info(f"[Sentinel-2] Skipping (Exists): {fname}")
             return

        try:
            geemap.ee_export_image(s2_bands, filename=out_path, scale=10, region=roi, file_per_band=False)
            logging.info(f"[Sentinel-2] Saved: {fname}")
        except Exception as e:
            logging.error(f"[Sentinel-2] Export failed: {e}")
