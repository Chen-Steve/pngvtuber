import numpy as np
import joblib

class ActionClassifier:
    def __init__(self, model_path="expression_model.pkl"):
        # load pkl
        self.bundle = joblib.load(model_path)
        self.model = self.bundle["model"]
        self.scaler = self.bundle["scaler"]
        self.label_encoder = self.bundle["label_encoder"]

    def predict(self, features):
        if features is None:
            return None
        X = np.array([features])
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)[0]
        label = self.label_encoder.inverse_transform([y_pred])[0]
        return label
