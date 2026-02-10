from sklearn.neural_network import MLPRegressor
import joblib
import os
import numpy as np
import logging

class ThermalAutoEncoder:
    def __init__(self, config):
        self.config = config
        self.params = config['models']['autoencoder']
        self.model_path = os.path.join(config['paths']['models'], 'autoencoder.pkl')
        
        # Ensure model dir exists
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        self.model = MLPRegressor(
            hidden_layer_sizes=tuple(self.params['hidden_layers']),
            activation='relu',
            solver='adam',
            max_iter=self.params['max_iter'],
            batch_size=self.params['batch_size'],
            learning_rate_init=self.params['learning_rate'],
            random_state=42,
            verbose=True
        )

    def train(self, X_train):
        logging.info("Training AutoEncoder...")
        self.model.fit(X_train, X_train) # AutoEncoder targets itself
        logging.info("Training complete.")

    def save(self):
        joblib.dump(self.model, self.model_path)
        logging.info(f"Model saved to {self.model_path}")

    def load(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            logging.info(f"Model loaded from {self.model_path}")
        else:
            logging.warning("Model file not found.")

    def get_reconstruction_error(self, X):
        X_pred = self.model.predict(X)
        mse = np.mean(np.square(X - X_pred), axis=1)
        return mse, X_pred
