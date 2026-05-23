import pandas as pd
from datetime import datetime
import os

LOG_FILE = "logs/mood_log.csv"

def log_mood(user, emotion, confidence, text):
    os.makedirs("logs", exist_ok=True)

    new_entry = pd.DataFrame([{
        "timestamp": datetime.now(),
        "user": user,
        "emotion": emotion,
        "confidence": confidence,
        "text": text
    }])

    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE)
        df = pd.concat([df, new_entry], ignore_index=True)
    else:
        df = new_entry

    df.to_csv(LOG_FILE, index=False)