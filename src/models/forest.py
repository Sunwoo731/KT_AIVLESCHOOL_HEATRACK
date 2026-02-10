from sklearn.ensemble import IsolationForest
import joblib
import os
import logging

class AnomalyIsolationForest:
    def __init__(self, config):
        self.config = config
        self.params = config['models']['isolation_forest']
        self.model_path = os.path.join(config['paths']['models'], 'isolation_forest.pkl')
        
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        self.model = IsolationForest(
            n_estimators=self.params['n_estimators'],
            contamination=self.params['contamination'],
            random_state=42,
            bootstrap=True
        )

    def train(self, X_train):
        logging.info("Training Isolation Forest...")
        self.model.fit(X_train)
        logging.info("Training complete.")

    def save(self):
        joblib.dump(self.model, self.model_path)
        logging.info(f"Model saved to {self.model_path}")

    def load(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
        else:
            logging.warning("Model file not found.")

    def predict_score(self, X):
        # Invert score so higher is more anomalous
        return -self.model.score_samples(X)
