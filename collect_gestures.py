import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

DATA_DIR = 'data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

gestures = ["Hello", "Yes", "No", "Thanks"]  # You can add more

for gesture in gestures:
    gesture_dir = os.path.join(DATA_DIR, gesture)
    os.makedirs(gesture_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    print(f"\n🖐️ Recording gesture: {gesture}")
    print("Press 'S' to start and 'Q' to stop recording.")

    with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
        recording = False
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

                    if recording:
                        data = []
                        for lm in landmarks.landmark:
                            data.extend([lm.x, lm.y, lm.z])

                        npy_path = os.path.join(gesture_dir, f"{count}.npy")
                        np.save(npy_path, np.array(data))
                        count += 1

            cv2.putText(frame, f"Gesture: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Collecting Gestures', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                print("▶️ Started recording...")
                recording = True
            elif key == ord('q'):
                print(f"🛑 Stopped recording for {gesture}. Saved {count} samples.")
                break

    cap.release()
    cv2.destroyAllWindows()
