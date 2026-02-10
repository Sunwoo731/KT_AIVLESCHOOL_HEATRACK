from ultralytics import YOLO
import numpy as np
import cv2
import os

class LeakSegmenter:
    def __init__(self, model_path=None):
        """
        Initialize YOLOv8-Seg model for leak pattern detection.
        If model_path is None, loads a pre-trained yolov8n-seg for demo.
        """
        if model_path and os.path.exists(model_path):
            self.model = YOLO(model_path)
        else:
            # Fallback to nano segment model
            self.model = YOLO('yolov8n-seg.pt') 
        
    def predict(self, image_patch):
        """
        image_patch: (H, W, 3) or (H, W) numpy array.
        Returns: 
            masks: List of binary masks (H, W)
            boxes: List of bounding boxes [x1, y1, x2, y2]
        """
        # Ensure 3-channel for YOLO
        if len(image_patch.shape) == 2:
            img_rgb = cv2.cvtColor((image_patch * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        elif image_patch.shape[2] == 1:
            img_rgb = cv2.cvtColor((image_patch[:,:,0] * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = (image_patch * 255).astype(np.uint8)

        # Run inference
        results = self.model(img_rgb, verbose=False)
        
        masks = []
        boxes = []
        
        for result in results:
            if result.masks is not None:
                for i, mask_data in enumerate(result.masks.data):
                    # Resize mask to original image size if needed
                    m = mask_data.cpu().numpy()
                    m_resized = cv2.resize(m, (image_patch.shape[1], image_patch.shape[0]))
                    masks.append(m_resized > 0.5)
                    boxes.append(result.boxes.xyxy[i].cpu().numpy())
                    
        return masks, boxes

    def calculate_leak_area(self, masks, pixel_resolution=10):
        """
        pixel_resolution: meters per pixel (default 10m for Sentinel-2 scale)
        Returns area in square meters.
        """
        total_pixels = 0
        for m in masks:
            total_pixels += np.sum(m)
            
        area_sq_m = total_pixels * (pixel_resolution ** 2)
        return area_sq_m

if __name__ == "__main__":
    # Self-test
    print("Testing LeakSegmenter...")
    seg = LeakSegmenter()
    dummy_img = np.random.rand(640, 640, 3)
    masks, boxes = seg.predict(dummy_img)
    print(f"Detected {len(masks)} objects.")
    area = seg.calculate_leak_area(masks)
    print(f"Estimated area: {area} m^2")
