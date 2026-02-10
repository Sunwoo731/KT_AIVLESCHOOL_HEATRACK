import os
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ee
import numpy as np
import rasterio
import requests
import pandas as pd
from datetime import datetime
from scipy.ndimage import uniform_filter
from tqdm import tqdm
import time
from utils.common import logger, create_directory
from utils.geo_tools import initialize_gee, get_roi_geometry, reproject_raster

def ee_to_wkt(ee_geom):
    """
    Converts GEE geometry to WKT string for CDSE OData.
    """
    try:
        coords = ee_geom.bounds().getInfo()['coordinates'][0]
        wkt_coords = ", ".join([f"{c[0]} {c[1]}" for c in coords])
        return f"POLYGON(({wkt_coords}))"
    except Exception as e:
        logger.error(f"Failed to convert EE geometry to WKT: {e}")
        return None

def safe_remove(filepath, retries=3, delay=1):
    """
    Attempts to remove a file with retries to handle Windows locking issues.
    """
    for i in range(retries):
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
            return True
        except PermissionError:
            time.sleep(delay)
    return False

class DataCollector:
    def __init__(self, config):
        self.cfg = config
        self.download_dir = self.cfg.RAW_DIR
        self.processed_dir = self.cfg.PROCESSED_DIR
        self.token = None
        self.token_expiry = 0
        create_directory(self.download_dir)
        create_directory(self.processed_dir)
        
        # Initialize GEE
        if not initialize_gee(self.cfg.GEE_PROJECT):
            raise ConnectionError("Failed to initialize Google Earth Engine")

    def get_auth_headers(self):
        """
        Returns auth headers, refreshing token if expired.
        """
        if self.token is None or time.time() > self.token_expiry:
            logger.info("Refreshing CDSE Access Token...")
            new_token = self._get_cdse_token()
            if new_token:
                self.token = new_token
                self.token_expiry = time.time() + 3000  # Proactive refresh at 50 mins
            else:
                return None
        return {"Authorization": f"Bearer {self.token}"}

    def _get_cdse_token(self):
        """
        Obtains an access token from CDSE Identity Provider.
        """
        data = {
            'client_id': 'cdse-public',
            'username': self.cfg.CDSE_USERNAME,
            'password': self.cfg.CDSE_PASSWORD,
            'grant_type': 'password',
        }
        try:
            r = requests.post("https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token", data=data, timeout=60)
            r.raise_for_status()
            return r.json()['access_token']
        except Exception as e:
            logger.error(f"Failed to get CDSE token: {e}")
            return None

    def _mask_s2_clouds(self, image):
        qa = image.select('QA60')
        cloud_bit_mask = 1 << 10
        cirrus_bit_mask = 1 << 11
        mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
        return image.updateMask(mask).divide(10000).copyProperties(image, ["system:time_start"])

    def _mask_l8_clouds(self, image):
        qa = image.select('QA_PIXEL')
        cloud_mask = (1 << 3)
        shadow_mask = (1 << 4)
        mask = qa.bitwiseAnd(cloud_mask).eq(0).And(qa.bitwiseAnd(shadow_mask).eq(0))
        
        # Separate scaling for SR and ST bands
        sr = image.select('SR_B.*').multiply(0.0000275).add(-0.2)
        st = image.select('ST_B.*').multiply(0.00341802).add(149.0)
        
        # Combine and apply mask
        return sr.addBands(st).updateMask(mask).copyProperties(image, ["system:time_start"])

    def refined_lee_filter(self, img_array, window_size=7):
        """
        Refined Lee Filter implementation for numpy array.
        """
        img_array = img_array.astype(np.float32)
        # Calculate local mean and variance
        mean = uniform_filter(img_array, size=window_size)
        sq_mean = uniform_filter(img_array**2, size=window_size)
        var = sq_mean - mean**2
        
        # Approximation: Use local variance as a proxy for heterogeneity
        overall_var = np.var(img_array)
        weights = var / (var + overall_var + 1e-10)
        filtered = mean + weights * (img_array - mean)
        return filtered.astype(np.float32)

    def get_sentinel2_data(self, roi, start_date, end_date):
        return (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                .filterBounds(roi)
                .filterDate(start_date, end_date)
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                .map(self._mask_s2_clouds)
                .select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12']))

    def get_landsat_data(self, roi, start_date, end_date):
        l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterBounds(roi).filterDate(start_date, end_date)
        l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2").filterBounds(roi).filterDate(start_date, end_date)
        return l8.merge(l9).map(self._mask_l8_clouds).select(['SR_B.*', 'ST_B.*'])

    def get_sentinel1_data(self, roi, start_date, end_date):
        return (ee.ImageCollection('COPERNICUS/S1_GRD')
                .filterBounds(roi)
                .filterDate(start_date, end_date)
                .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
                .filter(ee.Filter.eq('instrumentMode', 'IW')))

    def download_gee_collection(self, collection, prefix, roi, scale=10):
        """
        Downloads all images in compliance with the memory requirements.
        """
        img_list = collection.toList(collection.size())
        count = img_list.size().getInfo()
        logger.info(f"Downloading {count} images for {prefix}...")
        
        downloaded_files = []
        for i in range(count):
            img = ee.Image(img_list.get(i))
            date = ee.Date(img.get('system:time_start')).format('YYYYMMDD').getInfo()
            filename = f"{prefix}_{date}.tif"
            filepath = os.path.join(self.download_dir, filename)
            
            if os.path.exists(filepath):
                downloaded_files.append(filepath)
                continue

            try:
                # Select visual bands or all relevant bands
                # Use all bands present in the image (filtered by collection)
                bands = img.bandNames().getInfo()

                url = img.select(bands).getDownloadURL({
                    'scale': scale,
                    'crs': 'EPSG:3857',
                    'region': roi,
                    'format': 'GEO_TIFF'
                })
                
                r = requests.get(url, stream=True, timeout=60)
                with open(filepath, 'wb') as f:
                    for chunk in r.iter_content(1024):
                        f.write(chunk)
                downloaded_files.append(filepath)
            except Exception as e:
                logger.error(f"Failed to download {filename}: {e}")
        
        return downloaded_files

    def data_generator(self, file_list):
        """
        Yields data chunks (images) one by one for memory efficiency.
        """
        for fpath in file_list:
            if not os.path.exists(fpath):
                logger.warning(f"File not found: {fpath}. Skipping.")
                continue
            try:
                with rasterio.open(fpath) as src:
                    data = src.read()
                    # If S1, apply filter
                    if 'S1' in fpath:
                        data[0] = self.refined_lee_filter(data[0])
                    
                    # Convert to float16
                    yield data.astype(np.float16), fpath
            except Exception as e:
                logger.error(f"Failed to open/process {fpath}: {e}. Removing corrupted file.")
                safe_remove(fpath)
                continue

    def download_cdse_s3(self, roi_geom):
        """
        Downloads Sentinel-3 (SLSTR LST) data from CDSE using OData API.
        """
        # 1. Prepare WKT for ROI
        coords = roi_geom.bounds().coordinates().getInfo()[0]
        # Close the ring if not closed
        wkt_coords = ", ".join([f"{c[0]} {c[1]}" for c in coords])
        roi_wkt = f"POLYGON(({wkt_coords}))"
        
        # 2. Query
        base_url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
        start_str = f"{self.cfg.START_DATE}T00:00:00.000Z"
        end_str = f"{self.cfg.END_DATE}T23:59:59.999Z"
        
        # Filter for Sentinel-3 SLSTR Land Surface Temperature
        filter_query = (
            f"Collection/Name eq 'SENTINEL-3' and "
            f"Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/Value eq 'SL_2_LST___') and "
            f"ContentDate/Start ge {start_str} and "
            f"ContentDate/Start le {end_str} and "
            f"OData.CSC.Intersects(area=geography'SRID=4326;{roi_wkt}')"
        )
        
        query = {
            "$filter": filter_query,
            "$orderby": "ContentDate/Start asc",
            "$top": 1000   # API recommended max
        }
        
        headers = self.get_auth_headers()
        if not headers:
            logger.error("Initial CDSE Token acquisition failed.")
            return

        try:
            logger.info("Querying CDSE Catalogue...")
            products = []
            
            # Initial Request
            r = requests.get(base_url, params=query, headers=headers, timeout=60)
            r.raise_for_status()
            data = r.json()
            products.extend(data.get('value', []))
            logger.info(f"Retrieved {len(data.get('value', []))} items from page 1")

            # Pagination
            while '@odata.nextLink' in data:
                next_link = data['@odata.nextLink']
                logger.info("Fetching next page...")
                r = requests.get(next_link, headers=headers, timeout=60)
                r.raise_for_status()
                data = r.json()
                new_items = data.get('value', [])
                products.extend(new_items)
                logger.info(f"Retrieved {len(new_items)} items. Total so far: {len(products)}")
                
            logger.info(f"Found total {len(products)} Sentinel-3 products.")
        except Exception as e:
            logger.error(f"CDSE Query failed: {e}")
            return

        # 3. Download
        for p in products:
            prod_id = p['Id']
            name = p['Name']
            down_url = f"{base_url}({prod_id})/$value"
            fpath = os.path.join(self.download_dir, f"{name}.zip")
            
            if os.path.exists(fpath):
                # Check if it's a valid zip file
                import zipfile
                try:
                    if zipfile.is_zipfile(fpath):
                        logger.info(f"Skipping existing valid zip: {name}")
                        continue
                    else:
                        logger.warning(f"Found existing file {name} but it is invalid (partial download?). Re-downloading...")
                        safe_remove(fpath)
                except Exception as e:
                    logger.warning(f"Error checking file {name}: {e}. Re-downloading...")
                    safe_remove(fpath)
            
            # Retry loop for 401 scenarios
            for attempt in range(2):
                headers = self.get_auth_headers()
                if not headers:
                    logger.error("Failed to obtain headers for download.")
                    break

                logger.info(f"Downloading {name} (Attempt {attempt+1})...")
                try:
                    # Handle redirect manually to preserve Authorization header
                    r_init = requests.get(down_url, headers=headers, allow_redirects=False, timeout=60)
                    if r_init.status_code == 401:
                         logger.warning("Unauthorized (401). Refreshing token and retrying...")
                         self.token = None # Force refresh
                         continue

                    if r_init.status_code in [301, 302, 303, 307, 308]:
                        real_url = r_init.headers['Location']
                        logger.info(f"Redirecting to storage: {real_url[:50]}...")
                        dl_response = requests.get(real_url, headers=headers, stream=True, timeout=60)
                    else:
                        dl_response = r_init
                    
                    if dl_response.status_code == 401:
                         logger.warning("Unauthorized (401) on storage URL. Refreshing token and retrying...")
                         self.token = None
                         continue

                    with dl_response as r_down:
                        r_down.raise_for_status()
                        total_size = int(r_down.headers.get('content-length', 0))
                        
                        with open(fpath, 'wb') as f:
                            with tqdm(total=total_size, unit='B', unit_scale=True, desc=name) as pbar:
                                for chunk in r_down.iter_content(chunk_size=8192):
                                    f.write(chunk)
                                    pbar.update(len(chunk))
                    logger.info(f"Successfully downloaded {name}")
                    break # Success, exit retry loop
                except Exception as e:
                    logger.error(f"Failed to download {name}: {e}")
                    if 'dl_response' in locals(): dl_response.close()
                    safe_remove(fpath)
                    if attempt == 1: break # Final attempt failed

    def execute(self):
        logger.info("Starting Data Collection Pipeline...")
        roi_geom = get_roi_geometry(self.cfg.TARGET_LAT, self.cfg.TARGET_LON, self.cfg.ROI_BUFFER_METERS)
        
        # 1. Download GEE Data
        s2_files = self.download_gee_collection(
            self.get_sentinel2_data(roi_geom, self.cfg.START_DATE, self.cfg.END_DATE), "S2", roi_geom
        )
        l8_files = self.download_gee_collection(
            self.get_landsat_data(roi_geom, self.cfg.START_DATE, self.cfg.END_DATE), "Landsat", roi_geom
        )
        s1_files = self.download_gee_collection(
            self.get_sentinel1_data(roi_geom, self.cfg.START_DATE, self.cfg.END_DATE), "S1", roi_geom
        )
        
        # 2. CDSE Download (S3)
        self.download_cdse_s3(roi_geom)

        # 3. Processing (Generator test)
        logger.info("Processing downloaded data...")
        # Example of consuming the generator
        for data_chunk, fname in self.data_generator(s1_files):
             logger.info(f"Processed {fname} with shape {data_chunk.shape}")

        logger.info("Data Collection Pipeline Finished.")

if __name__ == "__main__":
    # Dynamic config loading for standalone execution
    import importlib.util
    import sys

    # Default to '20191215 정자동' if not specified
    if len(sys.argv) > 1:
        target_project = sys.argv[1]
    else:
        target_project = "20191215 정자동" 
    
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), target_project, "config.py")
    
    if not os.path.exists(config_path):
        print(f"Error: Config not found at {config_path}")
        sys.exit(1)

    spec = importlib.util.spec_from_file_location("config", config_path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    
    collector = DataCollector(cfg)
    collector.execute()
