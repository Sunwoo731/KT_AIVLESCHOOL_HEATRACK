import os
import cv2
import numpy as np
import pandas as pd
from anomaly_detect import AnomalyDetector
from object_filter import ObjectFilter
import matplotlib.pyplot as plt

def main_pipeline(
    image_path, 
    model_path='D:/temp_arirang/runs/arirang_multiclass_v1/weights/best.pt',
    output_dir='results'
):
    print(f"=== Starting Pipeline for {os.path.basename(image_path)} ===")
    
    # 1. Load Image
    if not os.path.exists(image_path):
        print(f"Error: Image {image_path} not found.")
        return
        
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 2. Anomaly Detection (Mocking embeddings for demo)
    print("Running Anomaly Detection...")
    detector = AnomalyDetector()
    # Mocking memory bank fit for demo purposes
    # In real usage, you load a pre-fitted model or fit it here
    # detector.fit(some_normal_data) 
    
    # Mocking anomaly map (Random + Center spot)
    h, w = img.shape[:2]
    anomaly_map = np.random.rand(h, w) * 0.3
    # Add fake anomaly
    cv2.circle(anomaly_map, (w//2, h//2), 50, 0.9, -1) 
    
    # 3. Object Filtering
    print("Running Object Filtering...")
    obj_filter = ObjectFilter(model_path=model_path)
    
    # Get filtered map
    final_map, road_mask, ignore_mask = obj_filter.filter_anomaly_map(anomaly_map, img_rgb)
    
    # 4. Visualization
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 4, 1)
    plt.imshow(img_rgb)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.imshow(anomaly_map, cmap='jet', vmin=0, vmax=1)
    plt.title("Raw Anomaly Score")
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    # Overlay Road Mask
    plt.imshow(road_mask, cmap='gray')
    plt.title("Road Mask (ROI)")
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    plt.imshow(final_map, cmap='jet', vmin=0, vmax=1)
    plt.title("Filtered Anomaly (Road Only)")
    plt.axis('off')
    
    out_file = os.path.join(output_dir, f"result_{os.path.basename(image_path)}")
    plt.savefig(out_file)
    plt.close()
    
    print(f"Result saved to {out_file}")

if __name__ == "__main__":
    # Test with a sample image from the dataset
    # Find a sample image
    sample_dir = "D:/temp_arirang/images/train"
    if os.path.exists(sample_dir):
        files = os.listdir(sample_dir)
        if files:
            sample_img = os.path.join(sample_dir, files[0])
            main_pipeline(sample_img)
    else:
        print("Dataset not found, waiting for training...")
