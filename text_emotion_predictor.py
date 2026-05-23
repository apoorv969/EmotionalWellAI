import joblib
import os

MODEL_PATH = os.path.join("models", "text_emotion_model_multilingual.pkl")
VECT_PATH = os.path.join("models", "vectorizer.pkl")

model = None
vectorizer = None

if os.path.exists(MODEL_PATH) and os.path.exists(VECT_PATH):
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECT_PATH)
else:
    print("⚠️ Warning: Model or vectorizer not found. Text emotion detection will not work.")

def predict_emotion(text):
    if model is None or vectorizer is None:
        return "neutral", 0.0

    vec = vectorizer.transform([text])
    emotion = model.predict(vec)[0]
    confidence = max(model.predict_proba(vec)[0]) * 100
    return emotion, round(confidence, 2)