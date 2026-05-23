# modules/emotion_text_nlp.py

from modules.language_detect import detect_language
from modules.translator import translate_to_english
from modules.text_emotion_predictor import predict_emotion
from textblob import TextBlob


def analyze_text_emotion(text: str) -> dict:
    """
    Detect emotion from text (multilingual support with translation)
    """

    # Detect language
    lang = detect_language(text)

    # Translate if not English
    translated_text = text
    if lang != "en":
        try:
            translated_text = translate_to_english(text)
        except Exception:
            translated_text = text

    # Sentiment analysis fallback
    blob = TextBlob(translated_text)
    polarity = blob.sentiment.polarity

    if polarity > 0.2:
        emotion = "happy"
        confidence = 80.0

    elif polarity < -0.2:
        emotion = "sad"
        confidence = 80.0

    else:
        try:
            emotion = predict_emotion(translated_text)
            confidence = 75.0
        except Exception:
            emotion = "neutral"
            confidence = 60.0

    return {
        "emotion": emotion,
        "confidence": confidence,
        "language": lang,
        "translated_text": translated_text
    }


# ✅ This is the function your backend expects
def detect_emotion_from_text(text: str):
    return analyze_text_emotion(text)