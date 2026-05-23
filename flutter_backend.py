from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
import threading
import pandas as pd
import os
import time
import numpy as np
import cv2
from deepface import DeepFace
from transformers import pipeline, MarianMTModel, MarianTokenizer
from langdetect import detect
from speechbrain.inference import EncoderClassifier
import torchaudio
import torch

# ==========================================
# GLOBALS
# ==========================================

MOOD_LOG_FILE = "mood_log.csv"
csv_lock = threading.Lock()
START_TIME = time.time()

# ==========================================
# INITIALIZE FASTAPI
# ==========================================

app = FastAPI(title="EmotionalWell AI Backend - Flutter Compatible")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Flutter mobile/web
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)

# ==========================================
# CSV SETUP & CLEANUP
# ==========================================

def cleanup_csv():
    try:
        if not os.path.exists(MOOD_LOG_FILE):
            df_init = pd.DataFrame(columns=[
                "timestamp","user","emotion","confidence","text","source","platform"
            ])
            df_init.to_csv(MOOD_LOG_FILE, index=False)
            return

        df = pd.read_csv(MOOD_LOG_FILE, engine="python")
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
        df = df.loc[:, ~df.columns.duplicated()]
        df["user"] = df.get("user", "Guest").fillna("Guest").astype(str).str.strip()
        df.loc[df["user"] == "", "user"] = "Guest"
        df.to_csv(MOOD_LOG_FILE, index=False)
    except Exception as e:
        print("CSV cleanup error:", e)

@app.on_event("startup")
def startup_cleanup():
    print("DEBUG: startup cleanup running")
    cleanup_csv()

# ==========================================
# Pydantic Models
# ==========================================

class TextData(BaseModel):
    text: str

class MoodLog(BaseModel):
    user: str
    emotion: str
    confidence: float
    text: Optional[str] = ""
    source: Optional[str] = "manual"
    platform: Optional[str] = "unknown"
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class FusionInput(BaseModel):
    text_emotion: Optional[str] = None
    text_confidence: Optional[float] = 0.0
    face_emotion: Optional[str] = None
    face_confidence: Optional[float] = 0.0
    voice_emotion: Optional[str] = None
    voice_confidence: Optional[float] = 0.0

# ==========================================
# TRANSLATION MODEL
# ==========================================

translator_model_name = "Helsinki-NLP/opus-mt-mul-en"
translator_tokenizer = MarianTokenizer.from_pretrained(translator_model_name)
translator_model = MarianMTModel.from_pretrained(translator_model_name)

def translate_to_english(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    try:
        detected_lang = detect(text)
        if detected_lang == "en":
            return text
        inputs = translator_tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        translated = translator_model.generate(**inputs, max_length=128, num_beams=5, do_sample=False)
        return translator_tokenizer.decode(translated[0], skip_special_tokens=True)
    except Exception as e:
        print("Translation error:", e)
        return text

# ==========================================
# TEXT EMOTION MODEL (unchanged)
# ==========================================

text_model = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)

EMOTION_MAP = {
    "admiration": "happy","amusement": "happy","approval": "happy",
    "joy": "happy","love": "happy","optimism": "happy",
    "gratitude": "happy","excitement": "happy",
    "anger": "anger","annoyance": "anger","disapproval": "anger",
    "fear": "fear","nervousness": "fear",
    "sadness": "sad","disappointment": "sad",
    "grief": "sad","remorse": "sad",
    "surprise": "surprise","realization": "surprise",
    "neutral": "neutral"
}

# ==========================================
# FACE EMOTION
# ==========================================

@app.post("/predict-face")
async def predict_face(file: UploadFile = File(...)):
    try:
        print(f"[Face API] Received file: filename={file.filename}, content_type={file.content_type}")
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            print("[Face API] ERROR: Could not decode image")
            return {"error": "Invalid image format", "status": 422}

        img = cv2.resize(img, (640, 480))
        result = DeepFace.analyze(img, actions=["emotion"], detector_backend="retinaface", enforce_detection=False, align=True)
        if isinstance(result, list):
            result = result[0]

        emotion = result.get("dominant_emotion", "neutral")
        confidence = float(result.get("emotion", {}).get(emotion, 0)) / 100

        print(f"[Face API] Detected emotion: {emotion}, confidence: {confidence}")
        return {"emotion": emotion, "confidence": round(confidence, 2)}
    except Exception as e:
        print(f"[Face API] Exception: {e}")
        return {"error": str(e), "status": 500}
# ==========================================
# VOICE EMOTION
# ==========================================

ser_model = EncoderClassifier.from_hparams(
    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
    savedir="tmp_ser_model"
)

VOICE_EMOTION_MAP = {
    "HAP": "happy",
    "SAD": "sad",
    "ANG": "angry",
    "NEU": "neutral",
    "FEA": "fear",
    "DIS": "disgust",
    "SUR": "surprise"
}

@app.post("/predict-voice")
async def predict_voice(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        tmp_file = f"temp_{int(time.time()*1000)}.wav"
        with open(tmp_file, "wb") as f:
            f.write(contents)

        signal, fs = torchaudio.load(tmp_file)
        if signal.shape[0] > 1:
            signal = signal.mean(dim=0, keepdim=True)
        signal = signal.unsqueeze(0)

        with torch.no_grad():
            prediction = ser_model.classify_batch(signal)
        if isinstance(prediction, dict):
            emotion_code = prediction.get("label", "NEU")
            confidence = float(prediction.get("score", 0.0))
        else:
            emotion_code = "NEU"
            confidence = 0.0

        emotion = VOICE_EMOTION_MAP.get(emotion_code, emotion_code.lower())
        confidence = min(max(confidence, 0.0), 1.0)

        if os.path.exists(tmp_file):
            os.remove(tmp_file)

        return {"emotion": emotion, "confidence": round(confidence, 2)}
    except Exception as e:
        print("Voice prediction failed:", e)
        return {"emotion": "neutral", "confidence": 0.0}

# ==========================================
# MULTIMODAL FUSION
# ==========================================

@app.post("/fusion")
async def multimodal_fusion(data: FusionInput):
    PRIORITY_ORDER = ["voice", "face", "text"]
    inputs = [
        (data.text_emotion, float(data.text_confidence or 0.0), "text"),
        (data.face_emotion, float(data.face_confidence or 0.0), "face"),
        (data.voice_emotion, float(data.voice_confidence or 0.0), "voice")
    ]
    inputs = [(e, c, m) for e, c, m in inputs if e]
    if not inputs:
        return {"final_emotion": "neutral", "overall_confidence": 0.0}

    weighted_scores = {}
    for emotion, confidence, modality in inputs:
        weighted_scores[emotion] = weighted_scores.get(emotion, 0.0) + confidence

    max_score = max(weighted_scores.values())
    top_emotions = [emo for emo, score in weighted_scores.items() if score == max_score]

    if len(top_emotions) > 1:
        final_emotion = top_emotions[0]
        for priority in PRIORITY_ORDER:
            for e, _, m in inputs:
                if e in top_emotions and m == priority:
                    final_emotion = e
                    break
            else:
                continue
            break
    else:
        final_emotion = top_emotions[0]

    contributing_confidences = [c for e, c, _ in inputs if e == final_emotion]
    overall_confidence = round(float(np.mean(contributing_confidences)), 2)
    return {"final_emotion": final_emotion, "overall_confidence": overall_confidence}

# ==========================================
# LOGGING, GET LOGS & HEALTH CHECK
# ==========================================

@app.post("/save-log")
async def save_log(log: MoodLog):
    safe_user = (log.user or "").strip() or "Guest"
    new_entry = {
        "timestamp": log.timestamp,
        "user": safe_user,
        "emotion": log.emotion,
        "confidence": log.confidence,
        "text": log.text,
        "source": log.source,
        "platform": log.platform
    }
    try:
        df = pd.read_csv(MOOD_LOG_FILE, engine="python")
    except Exception:
        df = pd.DataFrame(columns=new_entry.keys())
    df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    df = df.loc[:, ~df.columns.duplicated()]
    with csv_lock:
        df.to_csv(MOOD_LOG_FILE, index=False)
    return {"status": "saved"}

@app.get("/get-logs/{user}")
async def get_logs(user: str):
    try:
        df = pd.read_csv(MOOD_LOG_FILE, engine="python", on_bad_lines="skip")
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
        df = df.loc[:, ~df.columns.duplicated()]
        df["user"] = df["user"].astype(str).str.strip()
        user_logs = df[df["user"].str.lower() == user.strip().lower()]
        return user_logs.fillna("").to_dict(orient="records")
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/get-logs-all")
async def get_logs_all():
    try:
        df = pd.read_csv(MOOD_LOG_FILE, engine="python", on_bad_lines="skip")
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
        df = df.loc[:, ~df.columns.duplicated()]
        return df.fillna("").to_dict(orient="records")
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "message": "EmotionalWell AI Backend Running",
        "version": "WPR-14",
        "uptime_seconds": round(time.time() - START_TIME, 2)
    }

@app.get("/mobile-health")
def mobile_health():
    return {
        "status": "ok",
        "app": "EmotionalWell AI",
        "backend_version": "WPR-14"
    }