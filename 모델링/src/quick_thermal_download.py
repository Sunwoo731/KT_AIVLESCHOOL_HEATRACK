import os
import sys
import argparse
from src.data_collector import DataCollector, get_roi_geometry
from utils.common import setup_logger, logger

def quick_thermal_download(target_project):
    # Dynamic config loading
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.join(os.path.dirname(base_dir), target_project)
    config_path = os.path.join(project_dir, "config.py")
    
    import importlib.util
    spec = importlib.util.spec_from_file_location("config", config_path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    
    setup_logger()
    collector = DataCollector(cfg)
    roi_geom = get_roi_geometry(cfg.TARGET_LAT, cfg.TARGET_LON, cfg.ROI_BUFFER_METERS)
    
    logger.info(f"Targeting Thermal Only for {target_project}...")
    l8_files = collector.download_gee_collection(
        collector.get_landsat_data(roi_geom, cfg.START_DATE, cfg.END_DATE), "Landsat", roi_geom
    )
    logger.info(f"Downloaded {len(l8_files)} Landsat files.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("target", type=str)
    args = parser.parse_args()
    quick_thermal_download(args.target)
