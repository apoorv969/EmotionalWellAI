# ------------------------------------------------------------
# EmotionalWell AI Backend — WPR-14 REST Architecture
# Full Multilingual + Calibrated + Multimodal Fusion Version
# Developed by Apoorv Mittal
# ------------------------------------------------------------

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
print("🔥 RUNNING backend.py")   # ✅ ADD HERE
from deepface import DeepFace
import numpy as np
import cv2
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import os
from datetime import datetime
import time
from langdetect import detect
from speechbrain.inference import foreign_class
import threading

# ==========================================
# THREAD LOCK (NEW - WPR-14)
# ==========================================

csv_lock = threading.Lock()

# ==========================================
# CSV CLEANUP FUNCTION (AUTO + DEBUG)
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MOOD_LOG_FILE = os.path.join(BASE_DIR, "mood_log.csv")

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

ser_model = foreign_class(
    source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
    pymodule_file="custom_interface.py",
    classname="CustomEncoderWav2vec2Classifier"
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
# MULTILINGUAL TRANSLATION (FINAL WORKING FIX)
# ==========================================

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load model + tokenizer
translator_tokenizer = AutoTokenizer.from_pretrained(
    "facebook/nllb-200-distilled-600M"
)
translator_model = AutoModelForSeq2SeqLM.from_pretrained(
    "facebook/nllb-200-distilled-600M"
)

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
translator_model.to(device)


def translate_to_english(text: str) -> str:
    try:
        print("\n🔥 TRANSLATE FUNCTION CALLED")
        print("📝 INPUT:", text)

        # Handle empty input
        if not text or not text.strip():
            print("⚠️ Empty input")
            return text

        # 🔥 Force source language (French for now)
        translator_tokenizer.src_lang = "fra_Latn"

        # Tokenize
        inputs = translator_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 🔥 FIX: Correct way to get target language ID
        target_lang_id = translator_tokenizer.convert_tokens_to_ids("eng_Latn")

        # Generate translation
        translated_tokens = translator_model.generate(
            **inputs,
            forced_bos_token_id=target_lang_id,
            max_length=256
        )

        # Decode output
        translated_text = translator_tokenizer.batch_decode(
            translated_tokens,
            skip_special_tokens=True
        )[0]

        print("✅ TRANSLATED OUTPUT:", translated_text)

        return translated_text

    except Exception as e:
        print("❌ Translation Error:", e)
        return text
# ==========================================
# TEXT EMOTION MODEL
# ==========================================

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
@app.post("/predict-text")
async def predict_text(data: TextData):
    print("\n================ TEXT API CALLED ================")

    text = data.text.strip()
    print("📝 INPUT TEXT:", text)

    if not text:
        print("⚠️ Empty input received")
        return {
            "original_text": "",
            "translated_text": "",
            "emotion": "neutral",
            "confidence": 0.0
        }

    try:
        # 🔥 STEP 1: TRANSLATION
        print("\n🔄 Calling translate_to_english()...")
        translated_text = translate_to_english(text)

        print("✅ TRANSLATED TEXT:", translated_text)

        # 🔥 CHECK IF TRANSLATION FAILED
        if translated_text == text:
            print("⚠️ WARNING: Translation output SAME as input")

        # 🔥 STEP 2: EMOTION MODEL
        print("\n🤖 Running emotion model...")
        results = text_model(translated_text)[0]

        print("📊 RAW MODEL OUTPUT:", results)

        results = sorted(results, key=lambda x: x["score"], reverse=True)

        top_label = results[0]["label"]
        top_score = float(results[0]["score"])

        print(f"🏆 TOP LABEL: {top_label}, SCORE: {top_score}")

        emotion = EMOTION_MAP.get(top_label, "neutral")

        # 🔥 CONFIDENCE FILTERING
        if top_score < 0.30:
            print("⚠️ Low confidence → forcing neutral")
            emotion = "neutral"

        if len(results) > 1:
            second_score = float(results[1]["score"])
            if abs(top_score - second_score) < 0.03:
                print("⚠️ Ambiguous prediction → forcing neutral")
                emotion = "neutral"

        confidence = round(top_score, 2)

        print("🎯 FINAL EMOTION:", emotion)
        print("📈 FINAL CONFIDENCE:", confidence)

    except Exception as e:
        print("❌ ERROR in /predict-text:", e)

        translated_text = text
        emotion = "neutral"
        confidence = 0.0

    response = {
        "original_text": text,
        "translated_text": translated_text,
        "emotion": emotion,
        "confidence": confidence
    }

    print("\n🚀 FINAL API RESPONSE:", response)
    print("================================================\n")

    return response
# ==========================================
# FACE EMOTION
# ==========================================

@app.post("/predict-face")
async def predict_face(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return {"error": "Invalid image format"}

        img = cv2.resize(img, (640, 480))
        result = DeepFace.analyze(img, actions=["emotion"], enforce_detection=False)

        if isinstance(result, list):
            result = result[0]

        emotion = result.get("dominant_emotion", "neutral")
        confidence = float(result.get("emotion", {}).get(emotion, 0))

        return {"emotion": emotion, "confidence": confidence}

    except Exception as e:
        return {"error": str(e)}

# ==========================================
# VOICE EMOTION MAP (WPR-14 FIX)
# ==========================================

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

        # safer temporary filename (prevents conflicts)
        import time
        filename = f"temp_audio_{int(time.time()*1000)}.wav"

        with open(filename, "wb") as f:
            f.write(contents)

        prediction = ser_model.classify_file(filename)
        prediction_str = str(prediction)

        # Example: "HAP (100.0%)"
        emotion_code = prediction_str.split(" ")[0]

        confidence = float(
            prediction_str.split("(")[1].replace("%)", "")
        ) / 100

        # Convert model output to clean emotion
        emotion = VOICE_EMOTION_MAP.get(emotion_code, emotion_code.lower())

        # remove temp file after inference
        import os
        if os.path.exists(filename):
            os.remove(filename)

        return {
            "emotion": emotion,   # always clean (happy, sad, angry etc.)
            "confidence": round(confidence, 2)
        }

    except Exception as e:
        return {"error": str(e)}
# ==========================================
# MULTIMODAL FUSION (UPDATED - WPR-14)
# ==========================================

@app.post("/fusion")
async def multimodal_fusion(data: FusionInput):

    weighted_scores = {}
    modality_count = 0

    inputs = [
        (data.text_emotion, data.text_confidence),
        (data.face_emotion, data.face_confidence),
        (data.voice_emotion, data.voice_confidence)
    ]

    for emotion, confidence in inputs:

        if emotion and confidence is not None:
            modality_count += 1
            weighted_scores[emotion] = weighted_scores.get(emotion, 0) + confidence

    if not weighted_scores:
        return {
            "final_emotion": "neutral",
            "overall_confidence": 0.0
        }

    final_emotion = max(weighted_scores, key=weighted_scores.get)

    # divide by actual number of modalities used
    overall_confidence = round(
        weighted_scores[final_emotion] / modality_count, 2
    )

    return {
        "final_emotion": final_emotion,
        "overall_confidence": overall_confidence
    }
# ==========================================
# SAVE LOG (THREAD SAFE - WPR-14)
# ==========================================

@app.post("/save-log")
async def save_log(log: MoodLog):

    print("\n🔥 SAVE LOG API HIT")

    # ✅ Ensure correct file path
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(BASE_DIR, "mood_log.csv")

    print("📂 Saving to:", file_path)

    # ✅ Clean user
    safe_user = (log.user or "").strip() or "Guest"

    # ✅ Prepare entry
    new_entry = {
        "timestamp": str(log.timestamp),   # FIXED
        "user": safe_user,
        "emotion": log.emotion,
        "confidence": log.confidence,
        "text": log.text,
        "source": log.source,
        "platform": log.platform
    }

    print("🆕 New Entry:", new_entry)

    # ✅ Read existing CSV safely
    try:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            print("📊 Existing rows:", len(df))
        else:
            print("⚠️ File not found, creating new")
            df = pd.DataFrame(columns=new_entry.keys())

    except Exception as e:
        print("❌ CSV READ ERROR:", e)
        df = pd.DataFrame(columns=new_entry.keys())

    # ✅ Append new row
    df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    df = df.loc[:, ~df.columns.duplicated()]

    print("📊 Rows after insert:", len(df))

    # ✅ Save CSV safely
    try:
        with csv_lock:
            df.to_csv(file_path, index=False)
        print("💾 CSV WRITE SUCCESS")

    except Exception as e:
        print("❌ CSV WRITE ERROR:", e)
        return {"status": "error", "message": str(e)}

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