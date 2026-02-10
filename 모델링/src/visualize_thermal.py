import os
import glob
import rasterio
import matplotlib.pyplot as plt
import numpy as np
import argparse
from datetime import datetime

def visualize_thermal_changes(project_dir, output_file="thermal_analysis.png"):
    data_dir = os.path.join(project_dir, "data", "raw")
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        return

    # Find Landsat files
    landsat_files = sorted(glob.glob(os.path.join(data_dir, "Landsat_*.tif")))
    if not landsat_files:
        print("No Landsat files found. Waiting for download...")
        return

    print(f"Found {len(landsat_files)} Landsat files.")
    
    # Store (date, mean_temp, max_temp)
    dates = []
    means = []
    maxs = []
    
    # We'll also plot the first and last image as heatmaps
    first_img = None
    last_img = None
    first_date = None
    last_date = None

    for fpath in landsat_files:
        try:
            with rasterio.open(fpath) as src:
                # Identify Thermal Band (ST_B10)
                # We requested ['SR_B.*', 'ST_B.*']. 
                # Descriptions might help, or we assume ST_B10 is usually the last or specifically named.
                # Let's inspect descriptions if available, else assume last band if > 7?
                # Actually commonly ST_B10 is separated or we assume we loaded it.
                
                # Check descriptions for 'ST_B10'
                descriptions = src.descriptions
                thermal_idx = -1
                if descriptions:
                    for i, desc in enumerate(descriptions):
                        if desc and 'ST_B10' in desc:
                            thermal_idx = i
                            break
                
                # If not found by name, and we know we downloaded SR(1-7) + ST(10), it might be the last one.
                # Let's try to read the last band if explicit name not found, but print warning.
                if thermal_idx == -1:
                    # Heuristic: If we have multiple bands, Thermal is likely the one with Kelvin values (idx 0 is usually aerosol or blue).
                    # Actually standard L2 SP products: SR_B1..7, ST_B10. 
                    thermal_idx = src.count - 1 
                
                # Read Thermal Band
                # L2 ST is scaled: 0.00341802 * DN + 149.0 (Kelvin)
                # But check if GEE DataCollector already scaled it?
                # DataCollector.get_landsat_data -> .select(['SR_B.*', 'ST_B.*'])
                # DataCollector.download_gee_collection -> raw export.
                # Usually GEE exports raw DN values unless we explicitly .multiply() in the script.
                # Wait, I didn't see scaling in get_landsat_data for ST band in the fix.
                # I only added .select().
                # LANDSAT/LC08/C02/T1_L2 ST_B10 is in Kelvin * 10 or similar? 
                # Actually GEE documentation says: 
                # "ST_B10": Scale 0.00341802, Offset 149.0.
                
                raw_data = src.read(thermal_idx + 1) # 1-based index
                
                # Mask fill values (usually 0 or similar)
                mask = raw_data > 0
                if mask.sum() == 0: continue
                
                # Apply Scale Factor for L2 Surface Temperature
                # Kelvin = DN * 0.00341802 + 149.0
                kelvin = raw_data.astype(np.float32) * 0.00341802 + 149.0
                celsius = kelvin - 273.15
                
                # Mask out invalid values (e.g. < -50C or > 100C)
                valid_mask = (celsius > -50) & (celsius < 100) & mask
                
                if valid_mask.sum() == 0: continue
                
                valid_temps = celsius[valid_mask]
                
                # Parse date from filename
                # Landsat_YYYYMMDD.tif
                basename = os.path.basename(fpath)
                date_str = basename.split('_')[1].split('.')[0]
                dt = datetime.strptime(date_str, "%Y%m%d")
                
                dates.append(dt)
                means.append(np.mean(valid_temps))
                maxs.append(np.percentile(valid_temps, 99)) # 99th percentile to be robust to outliers
                
                if first_img is None:
                    first_img = celsius
                    first_date = dt
                last_img = celsius
                last_date = dt
                
        except Exception as e:
            print(f"Error processing {fpath}: {e}")

    # Plotting
    if not dates:
        print("No valid data points extracted.")
        return

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2)

    # 1. Time Series
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(dates, means, 'b-o', label='Mean Temp')
    ax1.plot(dates, maxs, 'r-x', label='Max Temp (99%)')
    ax1.set_title(f"Thermal Trend: {os.path.basename(project_dir)}")
    ax1.set_ylabel("Temperature (°C)")
    ax1.legend()
    ax1.grid(True)

    # 2. First Image
    ax2 = fig.add_subplot(gs[1, 0])
    im2 = ax2.imshow(first_img, cmap='inferno', vmin=np.percentile(first_img, 2), vmax=np.percentile(first_img, 98))
    ax2.set_title(f"Early State ({first_date.strftime('%Y-%m-%d')})")
    plt.colorbar(im2, ax=ax2, orientation='horizontal', label='°C')
    ax2.axis('off')

    # 3. Last Image
    ax3 = fig.add_subplot(gs[1, 1])
    im3 = ax3.imshow(last_img, cmap='inferno', vmin=np.percentile(first_img, 2), vmax=np.percentile(first_img, 98)) # Use same scale for comparison
    ax3.set_title(f"Recent State ({last_date.strftime('%Y-%m-%d')})")
    plt.colorbar(im3, ax=ax3, orientation='horizontal', label='°C')
    ax3.axis('off')

    output_path = os.path.join(project_dir, "data", "results", output_file)
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
        
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("project_path", type=str, help="Path to project folder")
    args = parser.parse_args()
    
    visualize_thermal_changes(args.project_path)
