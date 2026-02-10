import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import rasterio

class OverlayVisualizer:
    def __init__(self, save_dir="data/visuals/overlays"):
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

    def normalize_enhanced(self, array, percentile=98):
        p_low, p_high = np.percentile(array, [100-percentile, percentile])
        return np.clip((array - p_low) / (p_high - p_low), 0, 1)

    def create_advanced_overlay(self, rgb_img, thermal_img, hotspot_threshold=0.7, alpha=0.5, title="AI Heat Map Overlay"):
        print(f"Creating overlay for: {title}")
        # Apply enhanced normalization to RGB input
        rgb_enhanced = np.zeros_like(rgb_img)
        for i in range(3):
            rgb_enhanced[:,:,i] = self.normalize_enhanced(rgb_img[:,:,i])
        rgb_enhanced = np.clip(rgb_enhanced * 1.8, 0, 1) # Boost for context visibility

        plt.figure(figsize=(15, 15))
        
        # Base
        plt.imshow(rgb_enhanced)
        # Overlay
        im = plt.imshow(thermal_img, cmap='inferno', alpha=alpha, vmin=0.2, vmax=0.8)
        # Hotspots
        mask = (thermal_img > hotspot_threshold).astype(float)
        if mask.any():
            # Glowing contour effect
            plt.contour(mask, colors='white', linewidths=3, levels=[0.5], alpha=0.4)
            plt.contour(mask, colors='cyan', linewidths=1.5, levels=[0.5])

        plt.title(title, fontsize=18, fontweight='bold', color='white', backgroundcolor='black', pad=10)
        plt.axis('off')
        cbar = plt.colorbar(im, fraction=0.03, pad=0.04)
        cbar.set_label('Thermal Intensity (Risk Score)', fontsize=12)
        
        save_path = os.path.join(self.save_dir, "advanced_overlay.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        return save_path

        plt.title(title, fontsize=20, color='white', backgroundcolor='black')
        plt.axis('off')
        
        cbar = plt.colorbar(fraction=0.03, pad=0.04)
        cbar.set_label('Thermal Anomaly Intensity', fontsize=12)
        
        save_path = os.path.join(self.save_dir, "advanced_overlay.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=200, facecolor='black')
        plt.close()
        return save_path

def generate_visual_from_files(project_name="20191215 정자동"):
    raw_dir = f"{project_name}/data/raw"
    s2_path = os.path.join(raw_dir, "S2_201912348.tif")
    lst_path = os.path.join(raw_dir, "Landsat_201912349.tif")
    
    if not os.path.exists(s2_path) or not os.path.exists(lst_path):
        print(f"Required data for {project_name} not found.")
        return

    visualizer = OverlayVisualizer()
    
    with rasterio.open(s2_path) as src:
        print(f"S2 Band Count: {src.count}")
        # Load RGB
        red = visualizer.normalize_enhanced(src.read(3))
        green = visualizer.normalize_enhanced(src.read(2))
        blue = visualizer.normalize_enhanced(src.read(1))
        rgb = np.dstack((red, green, blue))
        print(f"RGB Shape: {rgb.shape}")
        
    with rasterio.open(lst_path) as src:
        print(f"LST Band Count: {src.count}")
        lst = visualizer.normalize_enhanced(src.read(1))
        print(f"LST Shape: {lst.shape}")
        
    # Generate Overlay
    save_path = visualizer.create_advanced_overlay(
        rgb, lst, 
        hotspot_threshold=0.75, 
        alpha=0.6, 
        title="Advanced Leak Detection Overlay: Bundang"
    )
    print(f"Advanced overlay saved to: {save_path}")

if __name__ == "__main__":
    generate_visual_from_files()
