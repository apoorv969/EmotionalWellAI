# modules/text_emotion_predictor.py
import os
import warnings
import joblib
from sklearn.dummy import DummyClassifier

# Path to your trained model
MODEL_PATH = os.path.join("models", "text_emotion_model.pkl")

# Load model safely
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        warnings.warn(f"Failed to load model at {MODEL_PATH}: {e}. Using dummy classifier instead.")
        model = DummyClassifier(strategy="uniform")
        model.fit([[0]], [0])
else:
    warnings.warn(f"Model file not found at {MODEL_PATH}. Using dummy classifier instead.")
    model = DummyClassifier(strategy="uniform")
    model.fit([[0]], [0])

def predict_emotion(text_features):
    """
    Predict the emotion label using the loaded model.
    text_features: preprocessed/text-vectorized data.
    """
    return model.predict([text_features])[0]
