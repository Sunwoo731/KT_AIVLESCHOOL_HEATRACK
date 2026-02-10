import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import os
import logging

class DashboardGenerator:
    def __init__(self, config):
        self.config = config
        self.output_dir = config['paths']['outputs']
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_report_image(self, lst_data, pipes_gdf, bounds, title="Analysis Report"):
        """Generate static map with pipe overlay."""
        msg = "Generating visual report..."
        logging.info(msg)
        
        # Setup Figure
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot Thermal
        vmin = np.percentile(lst_data, 2)
        vmax = np.percentile(lst_data, 98)
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
        
        im = ax.imshow(lst_data, cmap='jet', vmin=vmin, vmax=vmax, extent=extent, origin='upper')
        
        # Plot Pipes
        if pipes_gdf is not None:
             pipes_gdf.plot(ax=ax, color='cyan', linewidth=1.5, alpha=0.7, label='Heat Pipes')
        
        ax.set_title(title, fontsize=14)
        ax.axis('off')
        
        # Save
        filename = title.replace(" ", "_") + ".png"
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path, bbox_inches='tight', dpi=150)
        plt.close()
        logging.info(f"Saved report to {path}")
