import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Point
import random
import os
import logging

class PipeSimulator:
    def __init__(self, config):
        self.config = config
        self.output_path = config['paths']['simulated_pipes']
        self.locations = config['locations']
        
        # Ensure directory exists
        out_dir = os.path.dirname(self.output_path)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)

    def generate(self):
        logging.info("Generating synthetic pipe network...")
        features = []
        
        for loc in self.locations:
            name = loc['name']
            lon = loc.get('lon')
            lat = loc.get('lat')
            
            if lon is None or lat is None:
                continue

            # 1. Main Pipe (East-West approx 2km)
            # +0.01 deg is roughly 1km
            p1 = (lon - 0.01, lat)
            p2 = (lon + 0.01, lat)
            line_geom = LineString([p1, p2])
            
            features.append({
                'geometry': line_geom,
                'properties': {
                    'name': name,
                    'type': 'Main Distribution Pipe',
                    'status': 'Active',
                    'simulated': True
                }
            })
            
            # 2. Branch Pipe (Random)
            if random.choice([True, False]):
                p3 = (lon, lat - 0.005)
                p4 = (lon, lat + 0.005)
                line_geom_branch = LineString([p3, p4])
                features.append({
                    'geometry': line_geom_branch,
                    'properties': {
                        'name': name + "_Branch",
                        'type': 'Branch Pipe',
                        'status': 'Active',
                        'simulated': True
                    }
                })

        gdf = gpd.GeoDataFrame(features, crs="EPSG:4326")
        gdf.to_file(self.output_path, driver='GeoJSON')
        logging.info(f"Saved synthetic pipes to: {self.output_path}")

if __name__ == "__main__":
    # Test run
    # Mock config
    pass
