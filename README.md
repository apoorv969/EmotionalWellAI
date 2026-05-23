# EmotionalWell AI
- A personal companion for emotional awareness, reflection, and well-being.
- Track your emotions through webcam, text, and images while exploring insights, comparisons, and wellness tools.
## Features
1. Multi-Modal Emotion Detection
   - Webcam Analysis (DeepFace)
   - Image Upload Recognition
   - Text-based Sentiment Analysis (TextBlob)
2. Mood Tracking & Visualization
   - Personal Moodboard with history
   - Interactive Charts (Pie, Trend)
   - Exportable Mood Logs (CSV)
3. User Comparison
   - Compare emotions accross different users
   - Grouped Bar Charts for Emotion Distribution
4. Wellness Tools
   - Mindfulness Practices
   - Daily Affirmation Generator
   - Global Helplines for Mental Health Support 
   - Feedback Submissions with Ratings
5. Data Persistence
   - Emotions saved in 'mood_log.csv'
   - Feedback stored in 'feedback_log.csv'
## Usage Overview
1. Open the app in your browser after running it with Streamlit.
2. Navigate between tabs:
   - Webcam->Captures your all emotions in live.
   - Text-Emotion Detector->The user writes his/her thoughts and then the text analyses sentiments.
   - Image-Upload Option->User has to upload any picture of his/her for emotion detection in that particular photo.
   - My Moodboard->Explores the user's personal mood history.
   - Comparison Moodboard->Shows all the history of all the emotions captured till date along with all the timings and dates displayed.
   - Wellness Tools->Assessess all the affirmations, mindfulness practices and helplines.
3. Logs are automatically stored for tracking and reflection.
## Project Structure
   EmotionalWellAI
   - app.py # Main Streamlit app
   - mood_log.csv # User mood logs (auto-created)
   - feedback_log.csv # Feedback logs (auto-created)
   - README.md # Documentation
## How this App Works
1. Emotion Detection
   - DeepFace analyzes facial expressions to determine dominant emotion.
   - TextBlob evaluates sentiment polarity from user-written text.
2. Logging & Storage
   - Each entry includes timestamp, username, confidence score, and optional text.
3. Visualisation
   - Interactive tables
   - Pie charts for Emotion Distribution
   - Scatter plots for mood trends over time
4. Wellness Tools
   - Built-in mindfulness practices and grounding techniques.
   - Daily affirmation generator personalised with username.
   - Feedback Form for continuous improvement.
## Example Use Cases
 - Track emotions over time for self-reflection.
 - Compare moods across a group, team, or family.
 - Access wellness resources when feeling low.
 - Generate insights about emotional patterns.
## License
 - This project is licensed under the MIT License.
## Support Helplines
If any of the users is struggling emotionally, please seek professional support:
 - In India: AASRA- +91 9820466726
 - In United States of America (USA): National Suicide Prevention Lifeline- Dial 988
 - In United Kingdom (UK): Samaritans- Dial 116123
 - International: The user needs to find the helpline phone number on his/her own.

Note: EmotionalWell AI is not a medical or diagnostic tool, and this application is designed for self-reflection and wellness support only. For urgent needs, the user needs to contact professional healthcare services immediately.