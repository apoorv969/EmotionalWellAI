# modules/language_detect.py

from langdetect import detect, DetectorFactory

# Ensure reproducibility
DetectorFactory.seed = 0


def detect_language(text: str) -> str:
    """
    Detect the language of the given text.

    Args:
        text (str): Input text.

    Returns:
        str: ISO 639-1 language code (e.g., 'en', 'hi', 'fr').
             Defaults to 'en' if detection fails.
    """
    try:
        language = detect(text)
        return language
    except Exception:
        return "en"
