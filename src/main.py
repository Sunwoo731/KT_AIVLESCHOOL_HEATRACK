import argparse
import logging
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.config_loader import load_config, ensure_dirs
from data.synthetic import PipeSimulator
from data.downloader import SatelliteDownloader
from processing.downscaling import ThermalDownscaler
from models.autoencoder import ThermalAutoEncoder
from visualization.dashboard import DashboardGenerator

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="HEATTRACK: Satellite Thermal Anomaly Detection")
    parser.add_argument('mode', choices=['download', 'simulate', 'train', 'pipeline'], help="Execution Helper Mode")
    parser.add_argument('--config', default='configs/config.yaml', help="Path to config file")
    
    args = parser.parse_args()
    
    # Load Config
    config = load_config(args.config)
    ensure_dirs(config)

    if args.mode == 'download':
        logging.info("Starting Data Download...")
        downloader = SatelliteDownloader(config)
        for loc in config['locations']:
             downloader.download_for_location(loc['name'], loc['lat'], loc['lon'], loc['date'])
             
    elif args.mode == 'simulate':
        logging.info("Starting Pipe Simulation...")
        sim = PipeSimulator(config)
        sim.generate()
        
    elif args.mode == 'train':
        logging.info("Starting Model Training (AutoEncoder)...")
        # Placeholder for actual training loop integrated with data loading
        model = ThermalAutoEncoder(config)
        # train_data = load_data(...)
        # model.train(train_data)
        logging.info("Training feature implemented in separate script/notebook for now. See notebooks/")
        
    elif args.mode == 'pipeline':
        logging.info("Running Full Pipeline...")
        # 1. Simulate Pipes
        sim = PipeSimulator(config)
        sim.generate()
        
        # 2. Downscaling Demo (requires files to exist)
        # This is a placeholder for the logic connecting the pieces
        logging.info("Use 'download' mode first to fetch data.")
        logging.info("Pipeline logic would proceed to Downscaling -> Detection -> Reporting")

if __name__ == "__main__":
    main()
