import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from .utils import load_config

# Load the configuration
config = load_config('config.yaml')
threshold=config['ModelArchitecture']['threshold']


class CombinedModel:
    def __init__(self, preprocessing_pipeline, model, threshold=threshold):
        self.preprocessing_pipeline = preprocessing_pipeline
        self.model = model
        self.threshold = threshold
        self.data_storage = []  

    def predict(self, X):
        X_processed = self.preprocessing_pipeline.transform(X)
        y_pred_prob = self.model.predict_proba(X_processed)[:, 1]
        return (y_pred_prob >= self.threshold).astype(int)

    def predict_proba(self, X):
        X_processed = self.preprocessing_pipeline.transform(X)
        return self.model.predict_proba(X_processed)

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            joblib.dump(self, f)

    @staticmethod
    def load(filepath):
        with open(filepath, 'rb') as f:
            return joblib.load(f)

    def store_data(self, X, y):
        self.data_storage.append((X, y))

    def get_stored_data(self):
        X_data = np.concatenate([data[0] for data in self.data_storage], axis=0)
        y_data = np.concatenate([data[1] for data in self.data_storage], axis=0)
        return X_data, y_data

