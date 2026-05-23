import cv2
from deepface import DeepFace

def detect_emotion_from_webcam():
    """Capture a frame from webcam and detect emotion."""
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        return None, None

    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None, None

    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        # DeepFace v0.0.75+ returns a list
        if isinstance(result, list):
            result = result[0]
        emotion = result['dominant_emotion']
        confidence = result['emotion'][emotion]
        return emotion, confidence
    except Exception as e:
        print(f"Error detecting emotion from webcam: {e}")
        return None, None


def detect_emotion_from_image(image_path: str):
    """Detect emotion from an image file."""
    try:
        result = DeepFace.analyze(img_path=image_path, actions=['emotion'], enforce_detection=False)
        if isinstance(result, list):
            result = result[0]
        emotion = result['dominant_emotion']
        confidence = result['emotion'][emotion]
        return emotion, confidence
    except Exception as e:
        print(f"Error detecting emotion from image: {e}")
        return None, None