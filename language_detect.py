from langdetect import detect, DetectorFactory
DetectorFactory.seed=0
def detect_language(text):
    try:
        lang=detect(text)
        return lang if lang in ["en","hi","fr"] else "en"
    except:
        return "en"