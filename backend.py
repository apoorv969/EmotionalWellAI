# ------------------------------------------------------------
# EmotionalWell AI Backend — WPR-14 REST Architecture
# Full Multilingual + Calibrated + Multimodal Fusion Version
# Developed by Apoorv Mittal
# ------------------------------------------------------------

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from deepface import DeepFace
import numpy as np
import cv2
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import os
from datetime import datetime
import time
from langdetect import detect
from speechbrain.inference import EncoderClassifier
import threading

# ==========================================
# THREAD LOCK (NEW - WPR-14)
# ==========================================

csv_lock = threading.Lock()

# ==========================================
# CSV CLEANUP FUNCTION (AUTO + DEBUG)
# ==========================================

MOOD_LOG_FILE = "mood_log.csv"

def cleanup_csv():
    try:
        print("DEBUG: cleanup_csv -> loading CSV")
        df = pd.read_csv(MOOD_LOG_FILE, engine="python")

        print("DEBUG: rows before cleanup =", len(df))
        print(df.head())

        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
        df = df.loc[:, ~df.columns.duplicated()]
        df.columns = df.columns.str.strip()

        if "User" in df.columns:
            print("DEBUG: merging 'User' column into 'user'")
            df["user"] = df["User"].combine_first(df.get("user", ""))
            df = df.drop(columns=["User"])

        if "user" in df.columns:
            print("DEBUG: user column before fix")
            print(df["user"].head())

            df["user"] = df["user"].fillna("").astype(str).str.strip()
            df.loc[df["user"] == "", "user"] = "Guest"

            print("DEBUG: user column after fix")
            print(df["user"].head())
        else:
            print("DEBUG: user column missing -> creating")
            df["user"] = "Guest"

        df.to_csv(MOOD_LOG_FILE, index=False)

        print("DEBUG: cleanup complete, rows =", len(df))

    except Exception as e:
        print(f"DEBUG: cleanup failed -> {e}")

# ==========================================
# FastAPI App
# ==========================================

START_TIME = time.time()

app = FastAPI(
    title="EmotionalWell AI Backend",
    description="Multimodal Emotion Detection API (Text, Face, Voice, Multimodal)",
    version="WPR-14"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# NEW: Fusion Model
class FusionInput(BaseModel):
    text_emotion: Optional[str] = None
    text_confidence: Optional[float] = 0.0
    face_emotion: Optional[str] = None
    face_confidence: Optional[float] = 0.0
    voice_emotion: Optional[str] = None
    voice_confidence: Optional[float] = 0.0

# ==========================================
# LOAD SER MODEL
# ==========================================

ser_model = EncoderClassifier.from_hparams(
    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
    savedir="tmp_ser_model"
)


# ==========================================
# HEALTH CHECK
# ==========================================

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "message": "EmotionalWell AI Backend Running",
        "version": "WPR-14",
        "uptime_seconds": round(time.time() - START_TIME, 2)
    }

@app.get("/")
def root():
    return {"status": "Backend running"}

# ==========================================
# MOOD LOG FILE SETUP
# ==========================================

if not os.path.exists(MOOD_LOG_FILE):
    print("DEBUG: mood_log.csv missing -> creating new file")
    df_init = pd.DataFrame(columns=[
        "timestamp","user","emotion","confidence","text","source","platform"
    ])
    df_init.to_csv(MOOD_LOG_FILE, index=False)

# ==========================================
# MULTILINGUAL TRANSLATION (FINAL FIX)
# ==========================================

from transformers import MarianMTModel, MarianTokenizer
from langdetect import detect

# Load multilingual → English model
translator_model_name = "Helsinki-NLP/opus-mt-mul-en"

translator_tokenizer = MarianTokenizer.from_pretrained(translator_model_name)
translator_model = MarianMTModel.from_pretrained(translator_model_name)


def translate_to_english(text: str) -> str:
    """
    Reliable multilingual → English translation
    Works well for short phrases and idioms
    """

    text = text.strip()
    if not text:
        return ""

    try:
        detected_lang = detect(text)
        print(f"DEBUG detected_lang: {detected_lang}")

        # Skip translation if already English
        if detected_lang == "en":
            return text

        # Tokenize
        inputs = translator_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128
        )

        # Generate translation
        translated = translator_model.generate(
            **inputs,
            max_length=128,
            num_beams=5,
            do_sample=False
        )

        translated_text = translator_tokenizer.decode(
            translated[0],
            skip_special_tokens=True
        )

        print(f"DEBUG translated_text: {translated_text}")

        return translated_text

    except Exception as e:
        print("Translation error:", e)
        return text
# ==========================================
# TEXT EMOTION MODEL (Fixed & Multilingual)
# ==========================================

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from langdetect import detect

app = FastAPI()

# ==============================
# Text Emotion Pipeline
# ==============================
text_model = pipeline(
    "text-classification",
    model="SamLowe/roberta-base-go_emotions",
    top_k=None
)

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

# ==============================
# Input Model
# ==============================
class TextData(BaseModel):
    text: str

# ==============================
# Predict Text Emotion
# ==============================
@app.post("/predict-text")
async def predict_text(data: TextData):
    text = data.text.strip()
    if not text:
        return {"original_text": "", "translated_text": "", "emotion": "neutral", "confidence": 0.0}

    try:
        # Translate to English
        translated_text = translate_to_english(text)
        print(f"DEBUG Original: {text}")
        print(f"DEBUG Translated: {translated_text}")

        # Pipeline expects a list of strings
        results = text_model([translated_text])[0]
        # Sort by score descending
        results = sorted(results, key=lambda x: x["score"], reverse=True)

        top_label = results[0]["label"]
        top_score = float(results[0]["score"])

        # Map to generic emotion
        emotion = EMOTION_MAP.get(top_label, "neutral")
        confidence = round(top_score, 2)

        # Handle low confidence
        if confidence < 0.3:
            emotion = "neutral"
            confidence = 0.0

    except Exception as e:
        print(f"DEBUG Text prediction failed: {e}")
        translated_text = text
        emotion = "neutral"
        confidence = 0.0

    return {
        "original_text": text,
        "translated_text": translated_text,
        "emotion": emotion,
        "confidence": confidence
    }
# ==========================================
# FACE EMOTION (IMPROVED ACCURACY)
# ==========================================

@app.post("/predict-face")
async def predict_face(file: UploadFile = File(...)):

    try:
        contents = await file.read()

        np_arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return {"error": "Invalid image format"}

        # Improve detection reliability
        img = cv2.resize(img, (640, 480))

        result = DeepFace.analyze(
            img,
            actions=["emotion"],
            detector_backend="retinaface",   # ⭐ much more accurate
            enforce_detection=False,
            align=True
        )

        if isinstance(result, list):
            result = result[0]

        emotion = result.get("dominant_emotion", "neutral")

        confidence = float(
            result.get("emotion", {}).get(emotion, 0)
        ) / 100   # normalize to 0–1

        return {
            "emotion": emotion,
            "confidence": round(confidence, 2)
        }

    except Exception as e:
        return {"error": str(e)}

# ==========================================
# VOICE EMOTION MODEL (Fixed for SpeechBrain v1.0+)
# ==========================================

from fastapi import UploadFile, File
import torchaudio
import torch
import time
import os

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
        # Save uploaded audio
        contents = await file.read()
        tmp_file = f"temp_{int(time.time()*1000)}.wav"
        with open(tmp_file, "wb") as f:
            f.write(contents)

        # Load audio
        signal, fs = torchaudio.load(tmp_file)  # [channels, samples]
        if signal.shape[0] > 1:
            signal = signal.mean(dim=0, keepdim=True)  # convert to mono
        signal = signal.unsqueeze(0)  # [1, 1, samples] for batch

        # Run model
        with torch.no_grad():
            prediction = ser_model.classify_batch(signal)  # SpeechBrain v1.0+ batch inference
        print(f"DEBUG Raw Voice Prediction: {prediction}")

        # Handle dict output: {"label": ..., "score": ...}
        if isinstance(prediction, dict):
            emotion_code = prediction.get("label", "NEU")
            confidence = float(prediction.get("score", 0.0))
        elif isinstance(prediction, str):
            # legacy string output: "HAP (98.0%)"
            emotion_code = prediction.split(" ")[0]
            confidence = float(prediction.split("(")[1].replace("%)", "")) / 100
        else:
            # fallback
            emotion_code = "NEU"
            confidence = 0.0

        # Map to generic emotion
        emotion = VOICE_EMOTION_MAP.get(emotion_code, emotion_code.lower())
        confidence = min(max(confidence, 0.0), 1.0)  # clamp 0-1

        # Cleanup temp file
        if os.path.exists(tmp_file):
            os.remove(tmp_file)

        return {"emotion": emotion, "confidence": round(confidence, 2)}

    except Exception as e:
        print(f"DEBUG Voice prediction failed: {e}")
        return {"emotion": "neutral", "confidence": 0.0}
# ==========================================
# MULTIMODAL FUSION (IMPROVED WPR-14)
# ==========================================

@app.post("/fusion")
async def multimodal_fusion(data: FusionInput):
    """
    Combines text, face, and voice emotions using confidence-weighted voting.
    Tie-breakers handled by priority: voice > face > text
    """

    import numpy as np

    # Priority for tie-breaker when confidences are similar
    PRIORITY_ORDER = ["voice", "face", "text"]

    # Collect inputs as (emotion, confidence, modality)
    inputs = [
        (data.text_emotion, float(data.text_confidence or 0.0), "text"),
        (data.face_emotion, float(data.face_confidence or 0.0), "face"),
        (data.voice_emotion, float(data.voice_confidence or 0.0), "voice")
    ]

    # Remove None emotions
    inputs = [(e, c, m) for e, c, m in inputs if e]

    if not inputs:
        return {"final_emotion": "neutral", "overall_confidence": 0.0}

    # Weighted score aggregation
    weighted_scores = {}
    for emotion, confidence, modality in inputs:
        weighted_scores[emotion] = weighted_scores.get(emotion, 0.0) + confidence

    # Find max weighted score
    max_score = max(weighted_scores.values())
    top_emotions = [emo for emo, score in weighted_scores.items() if score == max_score]

    # Tie-breaker using modality priority
    if len(top_emotions) > 1:
        final_emotion = top_emotions[0]  # default
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

    # Overall confidence: average of contributing modalities for final emotion
    contributing_confidences = [c for e, c, _ in inputs if e == final_emotion]
    overall_confidence = round(float(np.mean(contributing_confidences)), 2)

    return {
        "final_emotion": final_emotion,
        "overall_confidence": overall_confidence
    }
# ==========================================
# SAVE LOG (THREAD SAFE - WPR-14)
# ==========================================

@app.post("/save-log")
async def save_log(log: MoodLog):

    safe_user = (log.user or "").strip()
    if not safe_user:
        safe_user = "Guest"

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

# ==========================================
# GET LOGS
# ==========================================

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

# ==========================================
# GET ALL LOGS
# ==========================================

@app.get("/get-logs-all")
async def get_logs_all():
    try:
        df = pd.read_csv(MOOD_LOG_FILE, engine="python", on_bad_lines="skip")
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
        df = df.loc[:, ~df.columns.duplicated()]
        return df.fillna("").to_dict(orient="records")

    except Exception as e:
        return {"status": "error", "message": str(e)}
# ==========================================
# MOBILE APP HEALTH CHECK (NEW - WPR-14)
# ==========================================

@app.get("/mobile-health")
def mobile_health():
    return {
        "status": "ok",
        "app": "EmotionalWell AI",
        "backend_version": "WPR-14"
    }