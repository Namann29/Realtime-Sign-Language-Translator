# SignSpeak: Real-Time Sign Language Translator

**SignSpeak** is a real-time sign language translation system built with **Python, Mediapipe, OpenCV, and Scikit-learn**.  
The project captures hand gestures using a webcam, processes them through Mediapipe’s landmark detection, and classifies them using a trained machine learning model. Recognized gestures are displayed on-screen and optionally converted to **speech output** for better accessibility.

## Features
- Real-time gesture recognition using Mediapipe  
- Custom gesture dataset collection via webcam  
- Machine learning model training using Scikit-learn  
- Instant prediction with high accuracy  
- Integrated text-to-speech (TTS) for audio output  
- Simple and modular Python scripts  
- Lightweight and easy to run on most systems  

## How It Works
1. **Collect Gestures:**  
   Run the following command to record gesture samples using your webcam:  
   ```bash
   python collect_gestures.py
   ```
   Each gesture is captured as a set of hand landmark coordinates.

2. **Train the Model:**  
   Train a gesture recognition model using:  
   ```bash
   python train_model.py
   ```
   The model is saved as `sign_model.pkl` for future predictions.

3. **Run Real-Time Translator:**  
   Start real-time prediction and translation with:  
   ```bash
   python realtime_predict.py
   ```
   Gestures are displayed on-screen and spoken aloud using text-to-speech.

## Tech Stack
- **Python 3.10+**  
- **Mediapipe** for hand tracking  
- **OpenCV** for video processing  
- **NumPy & Scikit-learn** for model training  
- **pyttsx3** for speech synthesis  

## Future Enhancements
- Support for more gestures and regional sign languages  
- Web and mobile deployment  
- Deep learning integration (CNN/LSTM models)  
- Sentence-level translation using NLP  
