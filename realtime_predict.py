import cv2
import mediapipe as mp
import numpy as np
import joblib
import pyttsx3

engine = pyttsx3.init()
engine.setProperty('rate', 160)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

model, gestures = joblib.load("sign_model.pkl")

cap = cv2.VideoCapture(0)

sound_on = True
last_spoken = ""

print("Camera started... Press 'q' to quit | Press 's' to toggle sound")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    cv2.rectangle(frame, (0, 0), (640, 60), (0, 100, 250), -1)
    cv2.putText(frame, "SIGN TRANSLATOR", (10, 40),
                cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            data = []
            for lm in hand.landmark:
                data.extend([lm.x, lm.y, lm.z])

            pred = model.predict([np.array(data)])
            gesture = gestures[int(pred[0])]

            cv2.rectangle(frame, (10, 80), (630, 160), (0, 200, 0), -1)
            cv2.putText(frame, f"Gesture: {gesture}", (20, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

            if sound_on and gesture != last_spoken:
                engine.say(gesture)
                engine.runAndWait()
                last_spoken = gesture

    cv2.putText(frame, f"[S] Sound: {'ON' if sound_on else 'OFF'}", (400, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Sign Translator", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('s'):
        sound_on = not sound_on
        print(f"🔊 Sound {'enabled' if sound_on else 'disabled'}")

cap.release()
cv2.destroyAllWindows()
