from fastapi import FastAPI
app = FastAPI(title="EmotionalWell AI Backend")

@app.get("/")
def home():
    return {"status": "Backend running"}