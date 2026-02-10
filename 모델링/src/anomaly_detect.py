import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import os

class AnomalyDetector:
    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors
        self.memory_bank = None
        self.nn_model = None
    
    def get_precipitation_data(self, roi, start_date, end_date):
        # Placeholder for real API call (e.g., KMA or OpenWeather)
        logger_placeholder = lambda x: print(f"[RainData] {x}")
        return pd.Series()
        
    def fit(self, normal_embeddings_list):
        """
        Build Memory Bank from multiple normal images.
        normal_embeddings_list: List of (H, W, Dim) arrays
        """
        all_patches = []
        for emb in normal_embeddings_list:
            # emb: (H, W, Dim)
            h, w, d = emb.shape
            flat_patches = emb.reshape(-1, d)
            all_patches.append(flat_patches)
        
        # Combine all normal patches
        self.memory_bank = np.concatenate(all_patches, axis=0)
        
        # Coreset subsampling (Simplified: Random subsampling for demo, 
        # in production use Greedy Coreset Selection)
        if len(self.memory_bank) > 5000:
            indices = np.random.choice(len(self.memory_bank), 5000, replace=False)
            self.memory_bank = self.memory_bank[indices]
            
        # Fit NN model for fast distance lookup
        self.nn_model = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm='auto')
        self.nn_model.fit(self.memory_bank)
        print(f"Memory bank built with {len(self.memory_bank)} features.")
        
    def detect_anomalies(self, embeddings, timestamps=None):
        """
        PatchCore Anomaly Score Calculation.
        embeddings: (H, W, Dim)
        Returns high-res anomaly map (H, W)
        """
        if self.nn_model is None:
            print("Error: Model not fitted. Run .fit() with normal data first.")
            h, w, _ = embeddings.shape
            return np.zeros((h, w))
            
        h, w, d = embeddings.shape
        query_patches = embeddings.reshape(-1, d)
        
        # Find nearest neighbors in memory bank
        distances, indices = self.nn_model.kneighbors(query_patches)
        
        # PatchCore Score = Max distance to nearest neighbor
        # (Simplified: average of k-nearest distances)
        avg_distances = np.mean(distances, axis=1)
        anomaly_map = avg_distances.reshape(h, w)
        
        return anomaly_map
        
    def apply_rain_filtering(self, scores, rain_data, timestamps):
        """
        Suppress anomaly scores if it rained recently.
        """
        # Logic: If recent rainfall > threshold, reduce scores by multiplier
        # For now, return as is
        return scores
        
    def save_heatmap(self, scores, output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.figure(figsize=(10, 8))
        plt.imshow(scores, cmap='jet')
        plt.colorbar(label='Anomaly Score')
        plt.title("PatchCore Anomaly Map")
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        print(f"Heatmap saved to {output_path}")
        
    def generate_report(self, scores, output_dir="data/results"):
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, "risk_report.txt")
        
        max_score = np.max(scores)
        mean_score = np.mean(scores)
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=== Geospatial Intelligence Risk Report ===\n")
            f.write(f"Timestamp: {pd.Timestamp.now()}\n")
            f.write(f"Algorithm: PatchCore (Distance-based)\n")
            f.write(f"Max Anomaly Score: {max_score:.4f}\n")
            f.write(f"Mean Anomaly Score: {mean_score:.4f}\n")
            f.write("-------------------------------------------\n")
            if max_score > 0.5: # Threshold placeholder
                f.write("WARNING: High thermal anomaly detected. Inspection recommended.\n")
            else:
                f.write("STATUS: Normal. No significant thermal anomalies found.\n")
        print(f"Report generated at {report_path}")
