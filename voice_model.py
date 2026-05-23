import tensorflow as tf
import pickle
import os

MODEL_PATH = "models/voice_emotion_model.h5"
ENCODER_PATH = "models/voice_emotion_labels.pkl"

def load_voice_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Voice emotion model file not found")

    if not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError("Voice label encoder not found")

    model = tf.keras.models.load_model(MODEL_PATH)

    with open(ENCODER_PATH, "rb") as f:
        encoder = pickle.load(f)

    return model, encoder
