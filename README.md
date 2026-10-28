# ArthroKNEE - Early Arthritis Screening by Multimodal Fusion
Student prototype - NIE Mysuru, Minor Project

Fuses three sources: **clinical questionnaire + wearable sensors (ESP32) + camera gait analysis (OpenCV)**
Stage 1: arthritis likely?  ->  Stage 2: OA or RA?  ->  Risk level (Low / Moderate / High)
10 languages: English, Hindi, Kannada, Tamil, Telugu, Malayalam, Marathi, Bengali, Gujarati, Punjabi.

## Run it (Windows)
1. Install Python 3.10+ from python.org (tick "Add Python to PATH").
2. Unzip this folder anywhere, then double-click **run_app.bat**.
   First launch installs packages (a few minutes). The app opens in your browser at http://localhost:8501.
   If the saved model can't be read on your machine, the app trains its own in about a minute (once).

Manual alternative:
    python -m pip install -r requirements.txt
    python -m streamlit run arthroknee_app.py

## Quick demo
Sidebar -> Virtual patient -> choose RA, severity 0.7 -> Generate -> open the "Fusion result" tab.

## Files
arthroknee_app.py - the app | arthro_core.py - features, data generator, models | translations.py - 10 languages
knee_vision.py - camera + QR helpers | train_model.py - rebuild dataset/models and figures | slides/ - figures

## Honest limits
* Trained on SYNTHETIC patients: metrics validate the pipeline, not clinical accuracy.
* Translations are machine-written drafts; the questionnaire is an original draft inspired by validated instruments.
* Not a medical device. Never enter real patient data into a shared or online copy.

## Put it online (free, Streamlit Community Cloud)
1. Upload the CONTENTS of this folder (so arthroknee_app.py is at the top level) to a new public GitHub repository.
2. share.streamlit.io -> Create app -> "Yup, I have an app" -> pick the repo, branch main, main file arthroknee_app.py.
3. Advanced settings -> Python version 3.12 -> Save -> Deploy.
Online mode is detected automatically: live webcam, retraining and dataset upload are switched off, and a public-demo
notice is shown. Never enter real patient data into a public copy.
