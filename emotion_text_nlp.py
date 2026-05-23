# emotion_text_nlp.py
# NLP-based Emotion Detection Module for EmotionalWell AI

from modules.language_detect import detect_language
from modules.translator import translate_to_english
from modules.text_emotion_predictor import predict_emotion
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download VADER lexicon if not already
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

def analyze_text_emotion(text: str) -> dict:
    """
    Analyze text using VADER + TextBlob for emotion classification.
    Returns dictionary with sentiment scores and a predicted emotion label.
    """
    lang=detect_language(text)
    translated_text=translate_to_english(text,lang)
    emotion,confidence=predict_emotion(translated_text)
    return{
        "language":lang,
        "translated_text":translated_text,
        "emotion":emotion,
        "confidence":confidence
    }
    if not text or text.strip() == "":
        return {"error": "Empty text provided."}

    # VADER sentiment analysis
    vader_scores = sia.polarity_scores(text)

    # TextBlob polarity
    blob = TextBlob(text)
    blob_polarity = blob.sentiment.polarity

    # Combine signals to decide emotion
    compound = vader_scores['compound']
    if compound >= 0.6 or blob_polarity > 0.5:
        emotion = "happy"
    elif compound <= -0.6 or blob_polarity < -0.5:
        emotion = "sad"
    elif vader_scores['neg'] > 0.6:
        emotion = "angry"
    elif vader_scores['neu'] > 0.8:
        emotion = "neutral"
    else:
        emotion = "mixed"

    return {
        "text": text,
        "vader": vader_scores,
        "textblob_polarity": blob_polarity,
        "predicted_emotion": emotion
    }


# Run as script for quick testing
if __name__ == "__main__":
    sample_texts = [
        "I am feeling so happy today!",
        "This is the worst day ever...",
        "I am okay, just normal.",
        "Why are you ignoring me? I'm angry!",
        "Life feels confusing sometimes."
    ]
    for t in sample_texts:
        result = analyze_text_emotion(t)
        print(f"Input: {t}")
        print(f"Predicted Emotion: {result['predicted_emotion']}")
        print("-" * 40)