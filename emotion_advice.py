def get_advice(emotion):
    """
    Returns advice/tips based on detected emotion
    """
    advice_dict = {
        "happy": "Keep up the good mood! Maybe share your happiness with a friend.",
        "sad": "It's okay to feel sad. Take a short walk or talk to someone you trust.",
        "neutral": "Maintain your balance. Maybe try a relaxing activity or mindfulness.",
        "angry": "Take deep breaths and calm down. Try listening to soothing music.",
        "surprise": "Embrace the moment and reflect on what surprised you.",
        "fear": "Acknowledge your fear and take small steps to feel safer."
    }
    return advice_dict.get(emotion.lower(), "Take care of yourself and stay mindful.")