# ------------------------------------------------------------
# EmotionalWell AI 🌈
# Developed by Apoorv Mittal
# ------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
from datetime import datetime

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import io

# ============================================================
# 🌐 MULTILINGUAL SYSTEM (GLOBAL - WPR 11 Compatible)
# ============================================================

language = st.sidebar.selectbox(
    "🌐 Select Language",
    ["English", "Hindi", "French"]
)

lang_map = {
    "English": "en",
    "Hindi": "hi",
    "French": "fr"
}

lang_code = lang_map.get(language, "en")

TRANSLATIONS = {
    "en": {
        "webcam_emotion": "Webcam Emotion",
        "text_emotion": "Text Emotion",
        "multimodal_emotion": "Multimodal Emotion",
        "image_emotion": "Image Emotion",
        "my_moodboard": "My Moodboard",
        "compare_moodboard": "Compare Moodboard",
        "voice_emotion": "Voice Emotion",
        "wellness_tools": "Wellness Tools",
        "enter_username": "Enter your username",
        "no_mood_data": "No mood data available.",
        "download_my_mood": "Download My Mood Data",
        "user_emotion_comparison": "User Emotion Comparison",
        "download_comparison": "Download Comparison Data",
        "support_text": "Your mental health matters. Take a pause and explore wellness tools below.",
        "mindfulness_expander": "Mindfulness Practices & Helplines",
        "daily_affirmation_expander": "Daily Affirmation Generator",
        "generate_affirmation": "Generate Affirmation"
    },
    "hi": {
        "webcam_emotion": "वेबकैम भावना",
        "text_emotion": "टेक्स्ट भावना",
        "multimodal_emotion": "मल्टीमॉडल भावना",
        "image_emotion": "छवि भावना",
        "my_moodboard": "मेरा मूडबोर्ड",
        "compare_moodboard": "मूड तुलना",
        "voice_emotion": "आवाज़ भावना",
        "wellness_tools": "वेलनेस टूल्स",
        "enter_username": "अपना उपयोगकर्ता नाम दर्ज करें",
        "no_mood_data": "कोई मूड डेटा उपलब्ध नहीं है।",
        "download_my_mood": "मेरा मूड डेटा डाउनलोड करें",
        "user_emotion_comparison": "उपयोगकर्ता भावना तुलना",
        "download_comparison": "तुलना डेटा डाउनलोड करें",
        "support_text": "आपका मानसिक स्वास्थ्य महत्वपूर्ण है। नीचे दिए गए वेलनेस टूल्स देखें।",
        "mindfulness_expander": "माइंडफुलनेस अभ्यास और हेल्पलाइन",
        "daily_affirmation_expander": "दैनिक सकारात्मक वाक्य",
        "generate_affirmation": "सकारात्मक वाक्य उत्पन्न करें"
    },
    "fr": {
        "webcam_emotion": "Émotion Webcam",
        "text_emotion": "Émotion Texte",
        "multimodal_emotion": "Émotion Multimodale",
        "image_emotion": "Émotion Image",
        "my_moodboard": "Mon Tableau d'Humeur",
        "compare_moodboard": "Comparer l'Humeur",
        "voice_emotion": "Émotion Voix",
        "wellness_tools": "Outils Bien-Être",
        "enter_username": "Entrez votre nom d'utilisateur",
        "no_mood_data": "Aucune donnée d'humeur disponible.",
        "download_my_mood": "Télécharger mes données d'humeur",
        "user_emotion_comparison": "Comparaison des émotions",
        "download_comparison": "Télécharger les données de comparaison",
        "support_text": "Votre santé mentale est importante. Explorez les outils de bien-être ci-dessous.",
        "mindfulness_expander": "Pratiques de pleine conscience et lignes d'assistance",
        "daily_affirmation_expander": "Générateur d'affirmation quotidienne",
        "generate_affirmation": "Générer une affirmation"
    }
}

def t(key):
    return TRANSLATIONS.get(lang_code, TRANSLATIONS["en"]).get(key, key)

# ============================================================
# FASTAPI BACKEND CONFIG (WPR-11 Architecture)
# ============================================================

BASE_URL = "http://127.0.0.1:8000"  # Change for Flutter testing if needed
SAVE_API = f"{BASE_URL}/save-log"
TEXT_API = f"{BASE_URL}/predict-text"
FACE_API = f"{BASE_URL}/predict-face"
IMAGE_API = f"{BASE_URL}/predict-face"
VOICE_API = f"{BASE_URL}/predict-voice"
FUSION_API = f"{BASE_URL}/fusion"
GET_LOG_API = f"{BASE_URL}/get-logs"
GET_LOG_ALL_API = "http://127.0.0.1:8000/get-logs-all"
# ============================================================
# EMOJI
# ============================================================

EMOJI = {
    "happy": "😊",
    "sad": "😔",
    "angry": "😡",
    "fear": "😟",
    "surprise": "😮",
    "neutral": "😐",
    "disgust": "🤢"
}
EMOTION_TRANSLATIONS = {
    "English": {
        "happy": "Happy",
        "sad": "Sad",
        "anger": "Anger",
        "fear": "Fear",
        "surprise": "Surprise",
        "neutral": "Neutral"
    },
    "Hindi": {
        "happy": "खुशी",
        "sad": "उदासी",
        "anger": "गुस्सा",
        "fear": "डर",
        "surprise": "आश्चर्य",
        "neutral": "सामान्य"
    },
    "French": {
        "happy": "Heureux",
        "sad": "Triste",
        "anger": "Colère",
        "fear": "Peur",
        "surprise": "Surprise",
        "neutral": "Neutre"
    }
}
# ============================================================
# LOAD MOOD DATA (FINAL STABLE FIX)
# ============================================================

REQUIRED_COLUMNS = [
    "timestamp",
    "user",
    "emotion",
    "confidence",
    "text",
    "source"
]

try:
    df = pd.read_csv("mood_log.csv", engine="python", on_bad_lines="skip")

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    # 🚨 REMOVE DUPLICATE COLUMNS (CRITICAL FIX)
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # Ensure required columns exist
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    # Reset index to avoid duplicate index errors
    df = df.reset_index(drop=True)

except Exception:
    df = pd.DataFrame(columns=REQUIRED_COLUMNS)

# Convert timestamp safely
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("EmotionalWell AI 🌈")
user_name = st.sidebar.text_input(t("enter_username"), value="Guest")
st.sidebar.markdown("---")
st.sidebar.markdown("Developed by Apoorv Mittal © 2026")

# ============================================================
# HELPERS
# ============================================================

def standardize_confidence(conf):
    try:
        conf = float(conf)
        return round(conf / 100, 3) if conf > 1 else round(conf, 3)
    except:
        return 0.0

def multimodal_decision_fusion(face=None, voice=None, text=None):
    sources = [s for s in [face, voice, text] if s]
    if not sources:
        return {"emotion":"neutral","confidence":0.0}
    return max(sources, key=lambda x: x["confidence"])

# ============================================================
# TABS
# ============================================================

tabs = st.tabs([
    f"📷 {t('webcam_emotion')}",
    f"📝 {t('text_emotion')}",
    f"🧩 {t('multimodal_emotion')}",
    f"🖼 {t('image_emotion')}",
    f"📊 {t('my_moodboard')}",
    f"📈 {t('compare_moodboard')}",
    f"🎤 {t('voice_emotion')}",
    f"🌿 {t('wellness_tools')}"
])

# ============================================================
# TAB 0: Webcam
# ============================================================

with tabs[0]:
    st.header(f"📷 {t('webcam_emotion')}")
    cam = st.camera_input("Capture Emotion")

    # --------------------------------------------------------
    # Webcam Analyze Button
    # --------------------------------------------------------
    if cam and st.button("Analyze Webcam Emotion"):
        try:
            files = {"file": cam.getvalue()}
            response = requests.post(FACE_API, files=files)

            if response.status_code == 200:
                result = response.json()

                emotion = result.get("emotion", "unknown")
                confidence = standardize_confidence(
                    result.get("confidence", 0)
                )

                translated_emotion = EMOTION_TRANSLATIONS.get(
                    language, {}
                ).get(emotion, emotion)

                st.success(
                    f"{translated_emotion} {EMOJI.get(emotion, '')} "
                    f"({confidence * 100:.1f}%)"
                )

                # 🔹 Save to backend instead of CSV
                from datetime import datetime

                log_payload = {
                    "timestamp": datetime.now().isoformat(),
                    "user": user_name,
                    "emotion": emotion,
                    "confidence": float(confidence),
                    "text": "",
                    "source": "face",
                    "platform": "webcam"
                }

                save_response = requests.post(SAVE_API, json=log_payload)

                if save_response.status_code == 200:
                    st.success("Mood saved successfully ✅")
                else:
                    st.warning(
                        f"Emotion detected but log saving failed.\n"
                        f"Status: {save_response.status_code}\n"
                        f"Response: {save_response.text}"
                    )

            else:
                st.error(
                    f"Backend API error. Status: {response.status_code}\n"
                    f"Response: {response.text}"
                )

        except Exception as e:
            st.error(f"API failed: {e}")
# ============================================================
# TAB 1: Text Emotion (WPR-12 Backend Save Version)
# ============================================================

with tabs[1]:

    st.header(f"📝 {t('text_emotion')}")

    user_text = st.text_area(
        "Write your thoughts...",
        height=150
    )

    if st.button("Analyze Text Emotion"):

        if not user_text.strip():
            st.warning("Please enter some text.")
        else:
            try:
                payload = {
                    "text": user_text
                }

                response = requests.post(TEXT_API, json=payload)

                if response.status_code == 200:

                    result = response.json()

                    # 🔹 Get translated text from backend
                    translated_text = result.get("translated_text", user_text)

                    emotion = result.get("emotion", "neutral")
                    confidence = standardize_confidence(
                        result.get("confidence", 0)
                    )

                    # 🔹 Show Translation FIRST
                    st.subheader("Translated to English:")
                    st.info(translated_text)

                    # 🔹 Then Show Emotion
                    selected_language = language

                    translated_emotion = EMOTION_TRANSLATIONS.get(
                        selected_language,
                        EMOTION_TRANSLATIONS["English"]
                    ).get(emotion, emotion)

                    st.success(
                        f"{translated_emotion} {EMOJI.get(emotion, '')} "
                        f"({confidence * 100:.1f}%)"
                    )

                    # ==================================================
                    # 🔥 NEW: SAVE TO FASTAPI BACKEND (NOT CSV)
                    # ==================================================

                    try:
                        save_response = requests.post(
                            SAVE_API,
                            json={
                                "timestamp": str(datetime.now()),
                                "user": user_name.strip() if user_name else "Guest",
                                "emotion": emotion.lower(),
                                "confidence": float(confidence),
                                "text": user_text,
                                "source": "text",
                                "platform": "streamlit"
                            }
                        )

                        if save_response.status_code == 200:
                            st.success("Mood saved successfully ✅")
                        else:
                            st.error("Failed to save mood log.")

                    except Exception as e:
                        st.error(f"Logging failed: {e}")

                else:
                    st.error("Backend API error.")

            except Exception as e:
                st.error(f"API failed: {e}")
# ======================================
# TAB 2: MULTIMODAL EMOTION DETECTION 🧩 (FINAL CLEAN VERSION)
# ======================================

with tabs[2]:

    st.header(f"🧩 {t('multimodal_emotion')}")

    cam = st.camera_input("Capture Face", key="mm_face")
    text_input = st.text_area("Enter Text", key="mm_text")
    voice_file = st.file_uploader("Upload Voice", type=["wav", "mp3"])

    if st.button("Analyze Multimodal Emotion"):

        st.subheader("🔄 Processing Multimodal Inputs...")

        face_data = None
        text_data = None
        voice_data = None

        # =============================
        # FACE API
        # =============================
        try:
            if cam is not None:
                files = {"file": cam.getvalue()}
                r = requests.post(FACE_API, files=files)

                if r.status_code == 200:
                    result = r.json()
                    face_data = {
                        "emotion": result.get("emotion", "neutral"),
                        "confidence": standardize_confidence(result.get("confidence", 0))
                    }
                else:
                    st.warning("Face API error.")
        except Exception:
            st.warning("Face API failed.")

        # =============================
        # TEXT API
        # =============================
        try:
            if text_input and text_input.strip():
                r = requests.post(TEXT_API, json={"text": text_input})

                if r.status_code == 200:
                    result = r.json()
                    text_data = {
                        "emotion": result.get("emotion", "neutral"),
                        "confidence": standardize_confidence(result.get("confidence", 0))
                    }
                else:
                    st.warning("Text API error.")
        except Exception:
            st.warning("Text API failed.")

        # =============================
        # VOICE API
        # =============================
        try:
            if voice_file is not None:
                files = {
                    "file": (
                        voice_file.name,
                        voice_file.getvalue(),
                        voice_file.type
                    )
                }

                r = requests.post(VOICE_API, files=files)

                if r.status_code == 200:
                    result = r.json()

                    voice_emotion = result.get("emotion", "neutral")
                    if isinstance(voice_emotion, list):
                        voice_emotion = voice_emotion[0] if voice_emotion else "neutral"

                    voice_data = {
                        "emotion": voice_emotion,
                        "confidence": standardize_confidence(result.get("confidence", 0))
                    }
                else:
                    st.warning("Voice API error.")

        except Exception:
            st.warning("Voice API failed.")

        # =============================
        # NO INPUT CHECK
        # =============================
        if not face_data and not text_data and not voice_data:
            st.warning("Please provide at least one input.")
            st.stop()

        # =============================
        # MODALITY SUMMARY
        # =============================
        st.markdown("---")
        st.subheader("📊 Modality Analysis Summary")

        summary_rows = []

        def add_row(name, data):
            if data:
                emotion = data["emotion"]
                translated = EMOTION_TRANSLATIONS.get(language, {}).get(
                    emotion, emotion
                )

                summary_rows.append({
                    "Modality": name,
                    "Emotion": translated,
                    "Confidence (%)": round(data["confidence"] * 100, 1)
                })

        add_row("Face", face_data)
        add_row("Text", text_data)
        add_row("Voice", voice_data)

        df_summary = pd.DataFrame(summary_rows)
        st.dataframe(df_summary, use_container_width=True)

        # =============================
        # CONFIDENCE CHART
        # =============================
        st.subheader("📈 Confidence Comparison")

        chart_data = df_summary.set_index("Modality")["Confidence (%)"]
        st.bar_chart(chart_data)

        # =============================
        # NORMALIZATION
        # =============================
        def normalize(data):
            if not data:
                return None, 0.0
            return data["emotion"], float(data["confidence"])

        text_emotion, text_conf = normalize(text_data)
        face_emotion, face_conf = normalize(face_data)
        voice_emotion, voice_conf = normalize(voice_data)

        # =============================
        # FUSION API
        # =============================
        fusion_payload = {
            "text_emotion": text_emotion,
            "text_confidence": text_conf,
            "face_emotion": face_emotion,
            "face_confidence": face_conf,
            "voice_emotion": voice_emotion,
            "voice_confidence": voice_conf
        }

        try:
            fusion_response = requests.post(FUSION_API, json=fusion_payload)

            if fusion_response.status_code == 200:
                fusion_result = fusion_response.json()
                final_emotion = fusion_result.get("final_emotion", "neutral")
                final_confidence = float(
                    fusion_result.get("overall_confidence", 0.0)
                )
            else:
                final_emotion = "neutral"
                final_confidence = 0.0

        except Exception:
            final_emotion = "neutral"
            final_confidence = 0.0

        # =============================
        # FINAL RESULT DISPLAY (FIXED)
        # =============================
        st.markdown("---")
        st.subheader("🎯 Final Multimodal Emotion")

        translated_final = EMOTION_TRANSLATIONS.get(language, {}).get(
            final_emotion, final_emotion
        )

        st.success(
            f"{translated_final} {EMOJI.get(final_emotion, '')} "
            f"({final_confidence * 100:.1f}%)"
        )

        # =============================
        # SAVE TO BACKEND
        # =============================
        log_payload = {
            "timestamp": datetime.now().isoformat(),
            "user": user_name if user_name else "Guest",
            "emotion": final_emotion,
            "confidence": float(final_confidence),
            "text": text_input.strip() if text_input else "",
            "source": "multimodal",
            "platform": "streamlit"
        }

        try:
            save_response = requests.post(SAVE_API, json=log_payload)

            if save_response.status_code == 200:
                st.success("Multimodal mood saved successfully ✅")
            else:
                st.warning("Emotion detected but saving failed.")

        except Exception:
            st.error("Backend connection error.")
# -----------------------------
# Tab 3: Image Emotion
# -----------------------------
with tabs[3]:

    st.header(f"🖼 {t('image_emotion')}")

    # -----------------------------
    # Language Selection
    # -----------------------------
    language = st.selectbox("Select Language", ["English", "Hindi", "French"])

    # -----------------------------
    # Upload Image
    # -----------------------------
    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file and st.button("Analyze Image Emotion"):

        try:
            # ✅ FIXED: Proper multipart file upload format
            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type
                )
            }

            response = requests.post(
                IMAGE_API,
                files=files
            )

            if response.status_code == 200:

                result = response.json()

                # ✅ Show real backend error if exists
                if "error" in result:
                    st.error(f"Backend error: {result['error']}")
                    st.stop()

                # -----------------------------
                # Extract emotion and confidence
                # -----------------------------
                emotion = result.get("emotion", "neutral")
                confidence = standardize_confidence(
                    result.get("confidence", 0)
                )

                # -----------------------------
                # Translation map
                # -----------------------------
                emotion_translation = {
                    "JOY": {"English": "JOY", "French": "Joie", "Hindi": "ख़ुशी"},
                    "SAD": {"English": "SAD", "French": "Tristesse", "Hindi": "दुःख"},
                    "FEAR": {"English": "FEAR", "French": "Peur", "Hindi": "भय"},
                    "ANGER": {"English": "ANGER", "French": "Colère", "Hindi": "गुस्सा"},
                    "SURPRISE": {"English": "SURPRISE", "French": "Surprise", "Hindi": "आश्चर्य"},
                    "DISGUST": {"English": "DISGUST", "French": "Dégoût", "Hindi": "घृणा"},
                    "NEUTRAL": {"English": "NEUTRAL", "French": "Neutre", "Hindi": "साधारण"},
                }

                emotion_translated = emotion_translation.get(
                    emotion.upper(), {}
                ).get(language, emotion)

                # -----------------------------
                # Display result
                # -----------------------------
                st.success(
                    f"{emotion_translated} "
                    f"{EMOJI.get(emotion, '')} "
                    f"({confidence * 100:.1f}%)"
                )

                # -----------------------------
                # SAVE TO BACKEND
                # -----------------------------
                log_payload = {
                    "timestamp": datetime.now().isoformat(),
                    "user": user_name.strip() if user_name else "Guest",
                    "emotion": emotion.lower(),
                    "confidence": float(confidence),
                    "text": "",
                    "source": "image",
                    "platform": "streamlit"
                }

                save_response = requests.post(
                    SAVE_API,
                    json=log_payload
                )
                if save_response.status_code != 200:
                    st.warning(
                        f"Emotion detected but log saving failed. "
                        f"Status: {save_response.status_code} "
                        f"Response: {save_response.text}"
                    )

            else:
                st.error(
                    f"Image analysis failed. "
                    f"Status: {response.status_code} "
                    f"Response: {response.text}"
                )

        except Exception as e:
            st.error(f"Error occurred: {str(e)}")
# -----------------------------
# Tab 4: My Moodboard (Production Version) – Fully Fixed
# -----------------------------
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime

# -----------------------------
# Ensure session state defaults
# -----------------------------
if "active_tab" not in st.session_state:
    st.session_state.active_tab = 0  # default tab (e.g., webcam)

if "moodboard_language" not in st.session_state:
    st.session_state.moodboard_language = "English"

# -----------------------------
# My Moodboard Tab – Fixed
# -----------------------------
with tabs[4]:
    st.header(f"📊 {user_name} - My Moodboard")

    # -----------------------------
    # LOAD DATA FROM BACKEND
    # -----------------------------
    try:
        api_url = f"{GET_LOG_API}/{user_name}"
        response = requests.get(api_url, timeout=(10, 60))
        response.raise_for_status()
        data = response.json()

        if not isinstance(data, list) or len(data) == 0:
            st.info(t("no_mood_data"))
            st.stop()

        df = pd.DataFrame(data)
        if df.empty:
            st.info(t("no_mood_data"))
            st.stop()

        df.columns = [str(c).strip().lower() for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated()]

    except Exception as e:
        st.error(f"Error loading mood data: {e}")
        st.stop()

    # -----------------------------
    # FILTER BY USER
    # -----------------------------
    user_col = next((col for col in df.columns if col.lower() == "user" or "user" in col.lower()), None)
    if user_col is None:
        st.error("No user column found in data.")
        st.stop()

    df[user_col] = df[user_col].astype(str).str.strip()
    user_df = df[df[user_col].str.lower() == str(user_name).strip().lower()].copy()
    if user_df.empty:
        st.info("No mood data available.")
        st.stop()

    # -----------------------------
    # TIMESTAMP & DATE
    # -----------------------------
    if "timestamp" not in user_df.columns:
        st.error("Missing timestamp column.")
        st.stop()

    user_df["timestamp"] = pd.to_datetime(user_df["timestamp"], errors="coerce")
    user_df = user_df.dropna(subset=["timestamp"])
    user_df["date"] = user_df["timestamp"].dt.date

    # -----------------------------
    # EMOTION NORMALIZATION (DROP UNKNOWN & remove Calm)
    # -----------------------------
    EMOTION_MAP = {
        "happy": "happy", "hap": "happy", "joy": "happy",
        "sad": "sad", "sadness": "sad",
        "neutral": "neutral", "neu": "neutral",
        "angry": "anger", "ang": "anger", "anger": "anger",
        "fear": "fear", "peur": "fear", "भय": "fear",
        "disgust": "disgust",
        "surprise": "surprise"  # Calm removed
    }

    user_df["emotion_normalized"] = user_df["emotion"].astype(str).str.lower().map(
        lambda x: EMOTION_MAP.get(x, None)
    )
    user_df = user_df.dropna(subset=["emotion_normalized"])
    if user_df.empty:
        st.info("No valid emotion data available after normalization.")
        st.stop()

    # -----------------------------
    # LANGUAGE SELECTION
    # -----------------------------
    if "moodboard_language" not in st.session_state:
        st.session_state.moodboard_language = "English"

    selected_language = st.selectbox(
        "Select Language",
        ["English", "Hindi", "French"],
        index=["English", "Hindi", "French"].index(st.session_state.moodboard_language),
        key="moodboard_language"
    )
    # ❌ Do NOT assign st.session_state.moodboard_language manually here

    EMOTION_DISPLAY_MAP = {
        "English": {
            "happy": "Happy",
            "sad": "Sad",
            "neutral": "Neutral",
            "anger": "Anger",
            "fear": "Fear",
            "disgust": "Disgust",
            "surprise": "Surprise"
        },
        "Hindi": {
            "happy": "खुश",
            "sad": "उदास",
            "neutral": "तटस्थ",
            "anger": "गुस्सा",
            "fear": "भय",
            "disgust": "घृणा",
            "surprise": "आश्चर्य"
        },
        "French": {
            "happy": "Heureux",
            "sad": "Triste",
            "neutral": "Neutre",
            "anger": "Colère",
            "fear": "Peur",
            "disgust": "Dégoût",
            "surprise": "Surprise"
        }
    }

    # Map normalized emotions to selected language
    user_df["emotion_display"] = user_df["emotion_normalized"].map(
        lambda x: EMOTION_DISPLAY_MAP[selected_language].get(x, x)
    )

    # -----------------------------
    # DATE SLIDER
    # -----------------------------
    from datetime import timedelta

    today = datetime.today().date()

    min_date_user = user_df["date"].min()
    max_date_user = max(user_df["date"].max(), today)

    # Fix: ensure min < max for Streamlit slider
    if min_date_user == max_date_user:
        max_date_user = min_date_user + timedelta(days=1)

    start_date, end_date = st.slider(
        "Select Date Range",
        min_value=min_date_user,
        max_value=max_date_user,
        value=(min_date_user, max_date_user),
        key="my_moodboard_date_slider"
    )
    # -----------------------------
    # EMOTION FILTER
    # -----------------------------
    unique_emotions = sorted(user_df["emotion_display"].unique())
    selected_emotions = st.multiselect(
        "Filter Emotions",
        options=unique_emotions,
        default=unique_emotions,
        key="my_moodboard_emotion_filter"
    )

    # -----------------------------
    # APPLY FILTERS
    # -----------------------------
    filtered_df = user_df[
        (user_df["date"] >= start_date) &
        (user_df["date"] <= end_date) &
        (user_df["emotion_display"].isin(selected_emotions))
    ]
    if filtered_df.empty:
        st.info("No mood data available for selected filters.")
        st.stop()

    # -----------------------------
    # EMOTION DISTRIBUTION (Histogram)
    # -----------------------------
    fig1 = px.histogram(
        filtered_df,
        x="emotion_display",
        color="emotion_display",
        text_auto=True,
        title="Emotion Distribution"
    )
    st.plotly_chart(fig1, use_container_width=True)

    # -----------------------------
    # EMOTION DISTRIBUTION (Pie Chart)
    # -----------------------------
    emotion_counts = filtered_df["emotion_display"].value_counts().reset_index()
    emotion_counts.columns = ["emotion", "count"]
    fig_pie = px.pie(
        emotion_counts,
        names="emotion",
        values="count",
        title="Emotion Proportion",
        hole=0.3
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    # -----------------------------
    # MOOD TREND
    # -----------------------------
    mood_trend = filtered_df.groupby("date").size().reset_index(name="count")
    fig2 = px.line(
        mood_trend,
        x="date",
        y="count",
        title="Mood Trend Over Time"
    )
    st.plotly_chart(fig2, use_container_width=True)

    # -----------------------------
    # DOWNLOAD
    # -----------------------------
    st.download_button(
        label=t("download_my_mood"),
        data=filtered_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{user_name}_mood.csv",
        mime="text/csv",
        key="my_moodboard_download"
    )
# -----------------------------
# Tab 5: Compare Moodboard (Multi-User Comparison) – Updated
# -----------------------------
with tabs[5]:
    st.header("📊 Compare Moodboard")

    try:
        api_url = GET_LOG_ALL_API
        response = requests.get(api_url)

        if response.status_code == 200:
            df = pd.DataFrame(response.json())
            df.columns = df.columns.str.strip().str.lower()
            df = df.loc[:, ~df.columns.duplicated()]
        else:
            st.error("Failed to fetch mood logs from backend.")
            df = pd.DataFrame()

    except Exception as e:
        st.error(f"Backend connection error: {e}")
        df = pd.DataFrame()

    # -----------------------------
    # Required columns
    # -----------------------------
    for col in ["timestamp", "user", "emotion"]:
        if col not in df.columns:
            df[col] = None

    for col in ["confidence", "text", "source"]:
        if col not in df.columns:
            df[col] = None

    # -----------------------------
    # Parse timestamp
    # -----------------------------
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])

    if df.empty:
        st.info("No mood data available.")
    else:
        df["user"] = df["user"].fillna("").astype(str).str.strip()
        df = df[df["user"] != ""]

        users = sorted(df["user"].unique().tolist())
        if len(users) == 0:
            st.info("No users found in data.")
        else:

            # -----------------------------
            # User selection
            # -----------------------------
            user1 = st.selectbox("Select User 1", users, key="comp_user1")
            df1 = df[df["user"] == user1]

            remaining_users = [u for u in users if u != user1]

            if len(remaining_users) > 0:
                user2 = st.selectbox("Select User 2", remaining_users, key="comp_user2")
                df2 = df[df["user"] == user2]
                df2_label = user2
            else:
                st.info("Only one user available — comparing sources instead.")
                df2 = df1.copy()
                df2_label = "source"

            # -----------------------------
            # Date range
            # -----------------------------
            all_dates = pd.concat([
                df1["timestamp"].dt.date,
                df2["timestamp"].dt.date
            ]).dropna().unique()

            today = datetime.today().date()

            if len(all_dates) > 0:
                min_date = min(all_dates)
                max_date = max(all_dates)
                max_date_for_slider = max(max_date, today)

                start_date, end_date = st.slider(
                    "Select Date Range",
                    min_value=min_date,
                    max_value=max_date_for_slider,
                    value=(min_date, max_date_for_slider),
                    key="compare_moodboard_date_slider"
                )
            else:
                start_date = end_date = today

            df1 = df1[
                (df1["timestamp"].dt.date >= start_date) &
                (df1["timestamp"].dt.date <= end_date)
            ]
            df2 = df2[
                (df2["timestamp"].dt.date >= start_date) &
                (df2["timestamp"].dt.date <= end_date)
            ]

            if df1.empty or df2.empty:
                st.info("No data for comparison in selected range.")
            else:

                # -----------------------------
                # Normalize emotions (Calm removed, Angry → Anger)
                # -----------------------------
                EMOTION_MAP = {
                    "happy": "happy",
                    "sad": "sad",
                    "neutral": "neutral",
                    "fear": "fear",
                    "peur": "fear",
                    "भय": "fear",
                    "angry": "anger",
                    "anger": "anger",
                    "disgust": "disgust",
                    "surprise": "surprise"  # Calm removed
                }

                df1["emotion"] = df1["emotion"].astype(str).str.lower().map(lambda x: EMOTION_MAP.get(x, None))
                df1 = df1.dropna(subset=["emotion"])

                df2["emotion"] = df2["emotion"].astype(str).str.lower().map(lambda x: EMOTION_MAP.get(x, None))
                df2 = df2.dropna(subset=["emotion"])

                # -----------------------------
                # Count emotions
                # -----------------------------
                df1_count = df1["emotion"].value_counts().reset_index()
                df1_count.columns = ["emotion", "count"]
                df1_count["label"] = user1

                df2_count = df2["emotion"].value_counts().reset_index()
                df2_count.columns = ["emotion", "count"]
                df2_count["label"] = df2_label

                combined_df = pd.concat([df1_count, df2_count])

                # -----------------------------
                # Bar Chart
                # -----------------------------
                st.subheader("📊 Emotion Comparison (Bar Chart)")
                fig = px.bar(
                    combined_df,
                    x="emotion",
                    y="count",
                    color="label",
                    barmode="group",
                    title="Emotion Comparison"
                )
                st.plotly_chart(fig, use_container_width=True)

                # -----------------------------
                # Pie Charts
                # -----------------------------
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader(f"🥧 {user1} Emotion Distribution")
                    fig_pie1 = px.pie(
                        df1_count,
                        names="emotion",
                        values="count",
                        title=f"{user1} Distribution"
                    )
                    st.plotly_chart(fig_pie1, use_container_width=True)

                with col2:
                    st.subheader(f"🥧 {df2_label} Emotion Distribution")
                    fig_pie2 = px.pie(
                        df2_count,
                        names="emotion",
                        values="count",
                        title=f"{df2_label} Distribution"
                    )
                    st.plotly_chart(fig_pie2, use_container_width=True)

                # -----------------------------
                # CSV download
                # -----------------------------
                st.download_button(
                    label="Download Comparison CSV",
                    data=combined_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"{user1}_comparison.csv",
                    mime="text/csv",
                    key="compare_moodboard_download"
                )
# ======================================
# TAB 6: ADVANCED VOICE EMOTION ANALYSIS 🎤
# ======================================

with tabs[6]:

    st.header(f"🎤 {t('voice_emotion')}")
    st.subheader("Choose Input Method")

    uploaded_file = st.file_uploader("📂 Upload WAV File", type=["wav"])
    recorded_audio = st.audio_input("🎙️ Record Your Voice")

    if st.button("Analyze Voice Emotion"):

        audio_bytes = None

        if recorded_audio is not None:
            audio_bytes = recorded_audio.getvalue()
            st.info("Using recorded audio.")
        elif uploaded_file is not None:
            audio_bytes = uploaded_file.getvalue()
            st.info("Using uploaded file.")
        else:
            st.warning("Please upload or record audio first.")
            st.stop()

        # =====================================
        # 📊 LOCAL ACOUSTIC FEATURE ANALYSIS
        # =====================================
        try:
            # Load audio
            y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)

            # Resample to 16kHz
            if sr != 16000:
                y = librosa.resample(y, orig_sr=sr, target_sr=16000)
                sr = 16000

            # Trim leading/trailing silence
            y, _ = librosa.effects.trim(y, top_db=20)

            # Limit to max 10 seconds
            max_len = sr * 10
            if len(y) > max_len:
                y = y[:max_len]

            duration = librosa.get_duration(y=y, sr=sr)
            st.subheader("📊 Acoustic Feature Dashboard")

            # Waveform
            fig_wave, ax_wave = plt.subplots()
            ax_wave.plot(y)
            ax_wave.set_title("Waveform")
            ax_wave.set_xlabel("Samples")
            ax_wave.set_ylabel("Amplitude")
            st.pyplot(fig_wave)

            # RMS Energy
            rms = librosa.feature.rms(y=y)[0]
            fig_rms, ax_rms = plt.subplots()
            ax_rms.plot(rms)
            ax_rms.set_title("Energy (RMS)")
            ax_rms.set_xlabel("Frame")
            ax_rms.set_ylabel("Energy")
            st.pyplot(fig_rms)
            intensity_score = float(np.mean(rms))
            st.info(f"Emotional Intensity Score: {intensity_score:.4f}")

            # Pitch (F0)
            f0, _, _ = librosa.pyin(
                y,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7")
            )
            f0_clean = f0[~np.isnan(f0)]
            if len(f0_clean) > 0:
                fig_pitch, ax_pitch = plt.subplots()
                ax_pitch.plot(f0_clean)
                ax_pitch.set_title("Pitch (F0)")
                ax_pitch.set_ylabel("Frequency (Hz)")
                ax_pitch.set_xlabel("Frame")
                st.pyplot(fig_pitch)
                mean_pitch = float(np.mean(f0_clean))
                pitch_std = float(np.std(f0_clean))
                st.info(f"Average Pitch: {mean_pitch:.2f} Hz")
                st.info(f"Voice Stability Index (Pitch Variability): {pitch_std:.2f}")
            else:
                st.warning("No pitch detected.")
                mean_pitch = 0
                pitch_std = 0

            # Spectrogram
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            fig_spec, ax_spec = plt.subplots()
            img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax_spec)
            ax_spec.set_title("Spectrogram (dB)")
            fig_spec.colorbar(img, ax=ax_spec, format="%+2.0f dB")
            st.pyplot(fig_spec)

            # Speaking Rate
            zero_crossings = librosa.zero_crossings(y)
            speaking_rate = np.sum(zero_crossings) / duration
            st.info(f"Speaking Activity Rate: {speaking_rate:.2f}")

        except Exception as e:
            st.warning(f"Feature extraction failed: {e}")

        # =====================================
        # 🎯 BACKEND VOICE EMOTION ANALYSIS
        # =====================================
        backend_failed = False
        emotion = "neutral"
        confidence = 0.0

        try:
            # Send preprocessed audio as 16kHz mono WAV
            buffer = io.BytesIO()
            sf.write(buffer, y, sr, format='WAV')
            buffer.seek(0)

            files = {"file": ("voice.wav", buffer, "audio/wav")}
            response = requests.post(VOICE_API, files=files, timeout=30)

            if response.status_code == 200:
                result = response.json()
                emotion = result.get("emotion", "neutral").lower()
                confidence = standardize_confidence(result.get("confidence", 0))
            else:
                backend_failed = True
        except Exception as e:
            backend_failed = True

        # =====================================
        # 🟢 LOCAL FALLBACK EMOTION ESTIMATION
        # =====================================
        if backend_failed or confidence < 0.1:
            # Simple heuristics based on pitch, energy, speaking rate
            if intensity_score > 0.06 and mean_pitch > 180:
                emotion = "happy"
            elif intensity_score < 0.03 and mean_pitch < 120:
                emotion = "sad"
            elif pitch_std > 50:
                emotion = "anger"
            elif speaking_rate > 3000:
                emotion = "surprise"
            elif intensity_score < 0.02:
                emotion = "neutral"
            else:
                emotion = "neutral"
            confidence = min(max(intensity_score + pitch_std/500, 0.1), 0.95)  # rough % estimate

        # Translation & emoji
        translated_emotion = EMOTION_TRANSLATIONS.get(language, {}).get(emotion, emotion)
        st.success(f"{translated_emotion.upper()} {EMOJI.get(emotion,'')} ({confidence*100:.1f}%)")
        st.progress(float(confidence))

        # Save to backend
        try:
            payload = {
                "timestamp": datetime.now().isoformat(),
                "user": user_name.strip() if user_name else "Guest",
                "emotion": emotion,
                "confidence": float(confidence),
                "text": "",
                "source": "voice",
                "platform": "streamlit"
            }
            save_response = requests.post(SAVE_API, json=payload)
            if save_response.status_code == 200:
                st.success("Voice mood saved to backend ✅")
            else:
                st.error("Failed to save mood to backend.")
        except Exception as e:
            st.error(f"Saving failed: {e}")
# -----------------------------
# Tab 7: Wellness Tools
# -----------------------------
with tabs[7]:
    st.header(f"🌿 {t('wellness_tools')}")
    st.markdown(f"<div style='background-color:#f0f9f9; padding:15px; border-radius:12px;'>{t('support_text')}</div>",
                unsafe_allow_html=True)

    # Mindfulness Practices & Helplines
    with st.expander(t("mindfulness_expander"), expanded=True):
        st.subheader("💡 Mindfulness Practices")
        practices = {
            "en": ["🌬 4-7-8 Breathing: Inhale 4s → Hold 7s → Exhale 8s",
                   "👣 5-4-3-2-1 Grounding: Identify 5 things you see, 4 you feel, 3 you hear, 2 you smell, 1 you taste",
                   "📝 Gratitude journaling: Write 3 things you are grateful for today"],
            "hi": ["🌬 4-7-8 श्वास: 4 सेकंड में साँस लें → 7 सेकंड रोकें → 8 सेकंड में छोड़ें",
                   "👣 5-4-3-2-1 ग्राउंडिंग: 5 चीजें देखें, 4 महसूस करें, 3 सुनें, 2 सूंघें, 1 चखें",
                   "📝 कृतज्ञता डायरी: आज के लिए 3 चीज़ें लिखें जिनके लिए आप आभारी हैं"],
            "fr": ["🌬 Respiration 4-7-8 : Inspirez 4s → Retenez 7s → Expirez 8s",
                   "👣 Ancrage 5-4-3-2-1 : Identifiez 5 choses que vous voyez, 4 que vous ressentez, 3 que vous entendez, 2 que vous sentez, 1 que vous goûtez",
                   "📝 Journal de gratitude : Écrivez 3 choses pour lesquelles vous êtes reconnaissant aujourd'hui"]
        }
        for p in practices[lang_code]:
            st.markdown(f"<span title='{p}'>{p}</span>", unsafe_allow_html=True)

        st.subheader("☎ Helplines")
        helplines = {
            "en": ["🇮🇳 India: AASRA - +91-9820466726", "🇺🇸 USA: National Suicide Prevention Lifeline - 988",
                   "🇬🇧 UK: Samaritans - 116 123", "🌍 International: Find a Helpline https://findahelpline.com"],
            "hi": ["🇮🇳 भारत: आश्रय - +91-9820466726", "🇺🇸 यूएसए: राष्ट्रीय आत्महत्या रोकथाम हेल्पलाइन - 988",
                   "🇬🇧 यूके: सामैरिटन्स - 116 123", "🌍 अंतर्राष्ट्रीय: हेल्पलाइन खोजें https://findahelpline.com"],
            "fr": ["🇮🇳 Inde : AASRA - +91-9820466726", "🇺🇸 USA : Ligne d'assistance prévention suicide - 988",
                   "🇬🇧 Royaume-Uni : Samaritans - 116 123",
                   "🌍 International : Trouver une ligne d'assistance https://findahelpline.com"]
        }
        for h in helplines[lang_code]:
            st.markdown(f"<span title='{h}'>{h}</span>", unsafe_allow_html=True)

        st.subheader("📚 Mindfulness Resources")
        resources = {
            "en": ["Calm", "Headspace", "Insight Timer"],
            "hi": ["Calm", "Headspace", "Insight Timer"],
            "fr": ["Calm", "Headspace", "Insight Timer"]
        }
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                f"<a href='https://www.calm.com' title='{resources[lang_code][0]}' target='_blank'>{resources[lang_code][0]}</a>",
                unsafe_allow_html=True)
        with col2:
            st.markdown(
                f"<a href='https://www.headspace.com' title='{resources[lang_code][1]}' target='_blank'>{resources[lang_code][1]}</a>",
                unsafe_allow_html=True)
        with col3:
            st.markdown(
                f"<a href='https://insighttimer.com' title='{resources[lang_code][2]}' target='_blank'>{resources[lang_code][2]}</a>",
                unsafe_allow_html=True)

    # Daily Affirmation Generator
    with st.expander(t("daily_affirmation_expander")):
        affirmations = {
            "en": [f"{user_name}, you are enough just as you are 🌸", "Every day is a fresh start, {user_name} 🌞",
                   "Focus on what you can control, {user_name} 🌱", "You are resilient and strong, {user_name} 💪"],
            "hi": [f"{user_name}, आप जैसी हैं वैसी ही पर्याप्त हैं 🌸", "हर दिन एक नई शुरुआत है, {user_name} 🌞",
                   "{user_name}, उस पर ध्यान दें जिसे आप नियंत्रित कर सकते हैं 🌱",
                   "आप मजबूत और लचीले हैं, {user_name} 💪"],
            "fr": [f"{user_name}, vous êtes suffisant tel que vous êtes 🌸",
                   "Chaque jour est un nouveau départ, {user_name} 🌞",
                   "Concentrez-vous sur ce que vous pouvez contrôler, {user_name} 🌱",
                   "Vous êtes résilient et fort, {user_name} 💪"]
        }
        if st.button(t("generate_affirmation") + " ✨"):
            st.success(np.random.choice(affirmations[lang_code]))
# -----------------------------
# Footer
# -----------------------------
st.markdown(
    "<center><small>EmotionalWell AI | Developed by Apoorv Mittal © 2026</small></center>",
    unsafe_allow_html=True
)