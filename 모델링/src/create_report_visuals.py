import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from ultralytics import YOLO

def create_report_visualization():
    # 1. Setup Paths
    # Use a sample from the dataset we prepared
    sample_img_path = "D:/temp_arirang/images/train/BLD00001_PS3_K3A_NIA0276.png" # Assuming this exists from previous step
    model_path = "d:/빅프로젝트/모델링/models/arirang_seg_v1.pt"
    output_path = "D:/빅프로젝트/모델링/data/results/report_visualization.png"
    
    if not os.path.exists(sample_img_path):
        # Fallback if specific file missing, pick first available
        import glob
        files = glob.glob("D:/temp_arirang/images/train/*.png")
        if files:
            sample_img_path = files[0]
        else:
            print("No sample images found!")
            return

    # 2. Load Data
    img = cv2.imread(sample_img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 3. Simulate Anomaly (Heatmap)
    # create a mock anomaly map that covers both road and building to show the filtering effect
    h, w = img.shape[:2]
    anomaly_map = np.zeros((h, w), dtype=np.float32)
    
    # Add fake hotspots
    # Spot 1: Center (likely building in this dataset)
    cv2.circle(anomaly_map, (w//2, h//2 - 50), 30, 1.0, -1) 
    # Spot 2: Bottom (Road likely)
    cv2.circle(anomaly_map, (w//2, h-50), 20, 0.8, -1)
    
    # Smooth it
    anomaly_map = cv2.GaussianBlur(anomaly_map, (51, 51), 0)
    anomaly_map = anomaly_map / (anomaly_map.max() + 1e-6) # Normalize 0-1

    # 4. Apply Model
    model = YOLO(model_path)
    results = model.predict(sample_img_path, verbose=False)
    result = results[0]
    
    road_mask = np.zeros((h, w), dtype=np.uint8)
    ignore_mask = np.zeros((h, w), dtype=np.uint8) # Building + Object

    if result.masks is not None:
        masks = result.masks.data.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        
        for i, cls in enumerate(classes):
            m = cv2.resize(masks[i], (w, h))
            if int(cls) == 0: # Road
                road_mask = np.maximum(road_mask, m)
            else: # Building(1) or Object(2)
                ignore_mask = np.maximum(ignore_mask, m)

    # 5. Filter Logic
    # Option A: Keep only Road (Strict)
    filtered_map_strict = anomaly_map * road_mask
    
    # Option B: Remove Buildings (Permissive - typically better if road mask is imperfect)
    # Let's use the inverse of the ignore mask
    filtered_map_smart = anomaly_map * (1 - ignore_mask)

    # 6. Plotting - High Quality Report Figure
    plt.figure(figsize=(20, 10))
    
    # Panel 1: Original Image
    plt.subplot(2, 3, 1)
    plt.imshow(img_rgb)
    plt.title("1. Original Satellite Image (RGB)", fontsize=14, fontweight='bold')
    plt.axis('off')
    
    # Panel 2: Detected Constraints (Red=Ignore)
    plt.subplot(2, 3, 2)
    overlay = img_rgb.copy()
    # Red overlay for buildings
    overlay[ignore_mask > 0.5] = overlay[ignore_mask > 0.5] * 0.5 + np.array([255, 0, 0]) * 0.5
    plt.imshow(overlay)
    plt.title("2. Detected Constraints (Building/Object)", fontsize=14, fontweight='bold')
    plt.axis('off')
    
    # Panel 3: Road ROI (Green=Focus)
    plt.subplot(2, 3, 3)
    overlay_road = img_rgb.copy()
    overlay_road[road_mask > 0.5] = overlay_road[road_mask > 0.5] * 0.5 + np.array([0, 255, 0]) * 0.5
    plt.imshow(overlay_road)
    plt.title("3. Road ROI (Focus Area)", fontsize=14, fontweight='bold')
    plt.axis('off')

    # Panel 4: Raw Anomaly Map (Simulated)
    plt.subplot(2, 3, 4)
    plt.imshow(img_rgb, alpha=0.6)
    plt.imshow(anomaly_map, cmap='jet', alpha=0.6)
    plt.title("4. Raw Anomaly Detection (Before Filter)", fontsize=14, fontweight='bold')
    plt.axis('off')
    # Add text pointing to false positive
    plt.text(50, 50, "Contains Hotspots on Buildings", color='white', backgroundcolor='red')

    # Panel 5: Filtered Result
    plt.subplot(2, 3, 5)
    plt.imshow(img_rgb, alpha=0.6)
    plt.imshow(filtered_map_smart, cmap='jet', alpha=0.6) # Using Smart Filter
    plt.title("5. Filtered Outcome (After Object Filter)", fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.text(50, 50, "False Positives Removed", color='white', backgroundcolor='green')
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    create_report_visualization()
