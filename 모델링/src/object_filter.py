from ultralytics import YOLO
import numpy as np
import cv2
import os

class ObjectFilter:
    def __init__(self, model_path='d:/빅프로젝트/모델링/models/arirang_seg_v1.pt'):
        """
        Initialize Arirang Multi-class Model.
        Classes: 0=Road, 1=Building, 2=Object
        """
        self.model_path = model_path
        if os.path.exists(model_path):
            print(f"Loading Object Filter Model: {model_path}")
            self.model = YOLO(model_path)
        else:
            print(f"WARNING: Model not found at {model_path}. Using fallback/nano.")
            self.model = YOLO('yolov8n-seg.pt') # Fallback
            
    def get_masks(self, image_rgb, target_size=None):
        """
        Returns:
            start_mask (road)
            stop_mask (building + object)
        """
        results = self.model(image_rgb, verbose=False)
        
        # Original size
        h, w = image_rgb.shape[:2]
        
        road_mask = np.zeros((h, w), dtype=np.uint8)
        ignore_mask = np.zeros((h, w), dtype=np.uint8) # Buildings + Objects
        
        for result in results:
            if result.masks is None:
                continue
                
            # data has (N, H, W) masks
            # cls has (N) class ids
            # boxes has (N, 4) if needed
            
            # Since masks are low-res usually, we should use 'xy' segments or upsample?
            # Ultralytics result.masks.data is torch tensor, resized to img size if we use result.masks.xy?
            # No, result.masks.data is usually smaller.
            # But result.plot() handles it.
            
            # Easier way: iterate and draw polygons? 
            # Or use result.masks.data (which checks size)
            
            # Let's iterate objects
            classes = result.boxes.cls.cpu().numpy()
            
            for i, cls_id in enumerate(classes):
                # Get binary mask for this object
                # Note: result.masks[i].data is (1, h_m, w_m)
                # Need to resize to (H, W)
                
                # Ultralytics convenience:
                # result.masks[i].xy contains polygon coordinates!
                poly = result.masks[i].xy[0].astype(np.int32)
                
                if cls_id == 0: # Road
                    cv2.fillPoly(road_mask, [poly], 1)
                elif cls_id == 1: # Building
                    cv2.fillPoly(ignore_mask, [poly], 1)
                elif cls_id == 2: # Object
                    cv2.fillPoly(ignore_mask, [poly], 1)
                    
        if target_size and target_size != (w, h):
            road_mask = cv2.resize(road_mask, target_size, interpolation=cv2.INTER_NEAREST)
            ignore_mask = cv2.resize(ignore_mask, target_size, interpolation=cv2.INTER_NEAREST)
            
        return road_mask, ignore_mask

    def filter_anomaly_map(self, anomaly_map, image_rgb):
        """
        Apply masking to anomaly map.
        anomaly_map: (H, W) float
        image_rgb: (H, W, 3) uint8
        """
        h, w = anomaly_map.shape
        road_mask, ignore_mask = self.get_masks(image_rgb, target_size=(w, h))
        
        # 1. Zero out ignored areas (Buildings/Objects)
        # 2. Keep only Roads (Optional: or just use Road mask as ROI)
        # User requirement: "Mask out non-road areas" -> means Keep Road.
        
        # Logic: 
        # Final = Anomaly * RoadMask * (1 - IgnoreMask)
        # Note: If RoadMask is accurate, (1 - IgnoreMask) is redundant but safe.
        
        filtered = anomaly_map * road_mask
        # filtered = filtered * (1 - ignore_mask) 
        
        return filtered, road_mask, ignore_mask
