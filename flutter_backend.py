# ------------------------------------------------------------
# EmotionalWell AI Backend - UPDATED LOGGING VERSION
# ------------------------------------------------------------
import os
import sys
import shutil
import io
import time
import threading
import datetime
import csv
import base64
import re

# 1. SILENCE TENSORFLOW WARNINGS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["HTTP_TIMEOUT"] = "300"

# 2. MANUALLY INJECT FFMPEG INTO PATH (Critical for Windows)
ffmpeg_path = r"C:\ffmpeg\bin"
if os.path.exists(ffmpeg_path):
    os.environ["PATH"] += os.pathsep + ffmpeg_path

import tensorflow as tf
import librosa
import pickle
import numpy as np
import cv2
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Any
from pydub import AudioSegment
from transformers import pipeline
from deepface import DeepFace
from deep_translator import GoogleTranslator

# 3. CONFIGURE PYDUB EXPLICITLY
AudioSegment.converter = r"C:\ffmpeg\bin\ffmpeg.exe"
AudioSegment.ffprobe = r"C:\ffmpeg\bin\ffprobe.exe"

# ==========================================
# GLOBALS & PATHS
# ==========================================
MOOD_LOG_FILE = "mood_log.csv"
csv_lock = threading.Lock()
START_TIME = time.time()

MODEL_DIR = "trained_text_emotion_model"
VOICE_MODEL_PATH = os.path.join(MODEL_DIR, "voice_emotion_model.h5")
VOICE_LABEL_PATH = os.path.join(MODEL_DIR, "voice_emotion_labels.pkl")

EMOTION_MAP = {
    "joy": "happy", "sadness": "sad", "anger": "angry",
    "fear": "fearful", "surprise": "surprised", "disgust": "disgusted",
    "happy": "happy", "sad": "sad", "fear": "fearful", "neutral": "neutral"
}

# ==========================================
# MODEL INITIALIZATION
# ==========================================
try:
    text_model = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=True
    )
    voice_model = tf.keras.models.load_model(VOICE_MODEL_PATH, compile=False)
    with open(VOICE_LABEL_PATH, "rb") as f:
        voice_encoder = pickle.load(f)
    print("  All AI Models loaded successfully.")
except Exception as e:
    print(f"  Warning: Model loading issues: {e}")

app = FastAPI(title="EmotionalWell AI Backend")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ==========================================
# DATA MODELS
# ==========================================
class TextData(BaseModel):
    text: str
    username: Optional[str] = "Guest"
    source: Optional[str] = "text_input"


class FaceData(BaseModel):
    image_base64: str
    username: Optional[str] = "Guest"
    source: Optional[str] = "face_input"


class FusionInput(BaseModel):
    username: str = "Guest"
    text_emotion: Optional[str] = None
    text_confidence: Any = 0.0
    face_emotion: Optional[str] = None
    face_confidence: Any = 0.0
    voice_emotion: Optional[str] = None
    voice_confidence: Any = 0.0


# ==========================================
# HELPERS
# ==========================================
def log_to_csv(emotion, confidence, source, username):
    """
    Saves entry in format: Timestamp, Emotion, Confidence, Source, User, Extra1, Extra2
    """
    try:
        with csv_lock:
            with open(MOOD_LOG_FILE, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.datetime.now(),
                    emotion,
                    confidence,
                    source,
                    username,
                    "",
                    ""
                ])
    except Exception as e:
        print(f"  CSV Error: {e}")


def translate_to_english(text: str) -> str:
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except Exception:
        return text


def clean_to_float(val):
    try:
        if val is None: return 0.0
        val_str = str(val).lower()
        if "tensor" in val_str:
            match = re.search(r"(\d+\.\d+)", val_str)
            num = float(match.group(1)) if match else 0.0
        else:
            num = float(val)
        if 0.0 < num <= 1.0: return round(num * 100, 2)
        return round(num, 2)
    except:
        return 0.0


# ==========================================
# ENDPOINTS
# ==========================================

@app.post("/predict-text")
async def predict_text(data: TextData):
    text = data.text.strip()
    username = data.username if data.username else "Guest"
    source = data.source if data.source else "text_input"

    if not text: return JSONResponse(content={"emotion": "neutral", "confidence": 0.0})
    try:
        translated = translate_to_english(text)
        results = text_model(translated)[0]
        results = sorted(results, key=lambda x: x["score"], reverse=True)
        top_res = results[0]

        conf = clean_to_float(top_res["score"])
        emotion = EMOTION_MAP.get(top_res["label"].lower(), "neutral")

        if conf < 30.0: emotion, conf = "neutral", 0.0

        # LOG DATA
        log_to_csv(emotion, conf, source, username)

        return JSONResponse(content={
            "emotion": emotion,
            "confidence": conf,
            "translated_text": translated
        })
    except Exception as e:
        print(f" Text Error: {e}")
        return JSONResponse(content={"emotion": "neutral", "confidence": 0.0})


@app.post("/predict-face")
async def predict_face(
        file: Optional[UploadFile] = File(None),
        username: str = Form("Guest"),
        source: str = Form("face_input")
):
    try:
        if not file: return JSONResponse(content={"emotion": "neutral", "confidence": 0.0})

        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        analysis = DeepFace.analyze(img_path=img, actions=['emotion'], enforce_detection=False)
        res = analysis[0] if isinstance(analysis, list) else analysis
        label = max(res['emotion'], key=res['emotion'].get)

        conf = round(float(res['emotion'][label]), 2)
        emotion = EMOTION_MAP.get(label.lower(), label.lower())

        # LOG DATA
        log_to_csv(emotion, conf, source, username)

        return JSONResponse(content={"emotion": emotion, "confidence": conf})
    except Exception as e:
        print(f" Face Error: {e}")
        return JSONResponse(content={"emotion": "neutral", "confidence": 0.0})


@app.post("/predict-voice")
async def predict_voice(
        file: UploadFile = File(...),
        username: str = Form("Guest"),
        source: str = Form("voice_input")
):
    try:
        audio_data = await file.read()
        audio_stream = io.BytesIO(audio_data)
        audio = AudioSegment.from_file(audio_stream).set_frame_rate(22050).set_channels(1)
        samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0

        mfccs = librosa.feature.mfcc(y=samples, sr=22050, n_mfcc=40)
        mfccs = np.pad(mfccs, ((0, 0), (0, max(0, 174 - mfccs.shape[1]))))[:, :174]
        features = np.expand_dims(np.expand_dims(mfccs, axis=0), -1)

        prediction = voice_model.predict(features)
        probs = prediction[0].tolist()
        best_idx = int(np.argmax(probs))

        final_conf = clean_to_float(probs[best_idx])
        labels = {0: "angry", 1: "disgust", 2: "fear", 3: "happy", 4: "neutral", 5: "sad", 6: "surprise"}
        emotion = labels.get(best_idx, "neutral")

        # LOG DATA
        log_to_csv(emotion, final_conf, source, username)

        return JSONResponse(content={"emotion": emotion, "confidence": final_conf})
    except Exception as e:
        print(f" VOICE ERROR: {e}")
        return JSONResponse(content={"emotion": "neutral", "confidence": 0.0})


@app.post("/fusion")
async def fusion(data: FusionInput):
    try:
        emotions = []
        inputs = [
            (data.text_emotion, clean_to_float(data.text_confidence)),
            (data.face_emotion, clean_to_float(data.face_confidence)),
            (data.voice_emotion, clean_to_float(data.voice_confidence))
        ]

        for emo, conf in inputs:
            if emo and str(emo).lower() != "neutral" and conf > 5.0:
                emotions.append((str(emo).lower(), conf))

        if not emotions:
            return JSONResponse(content={"final_emotion": "neutral", "overall_confidence": 0.0})

        scores = {}
        for emo, conf in emotions:
            scores[emo] = scores.get(emo, 0) + conf

        winner = max(scores, key=scores.get)
        avg_conf = scores[winner] / float(len(emotions))

        # LOG DATA
        log_to_csv(winner, avg_conf, "fusion_input", data.username)

        return JSONResponse(content={"final_emotion": str(winner), "overall_confidence": round(avg_conf, 2)})
    except Exception as e:
        return JSONResponse(content={"final_emotion": "neutral", "overall_confidence": 0.0})


@app.get("/get-logs-all")
async def get_logs_all():
    try:
        if not os.path.exists(MOOD_LOG_FILE): return []
        # Return raw CSV data as list of dictionaries
        df = pd.read_csv(MOOD_LOG_FILE, names=["timestamp", "emotion", "confidence", "source", "user", "ex1", "ex2"])
        return df.fillna("").to_dict(orient="records")
    except Exception:
        return []


@app.get("/health")
async def health(): return {"status": "ok", "uptime": round(time.time() - START_TIME, 2)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8001)
