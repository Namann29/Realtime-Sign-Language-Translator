import os
import cv2
import yt_dlp
import numpy as np
import mediapipe as mp

# ========== SETUP ==========
mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1)
DATA_DIR = "gestures"
VIDEOS_DIR = "yt_videos"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VIDEOS_DIR, exist_ok=True)

# ========== INPUT ==========
gesture_name = input("Enter gesture/word name (e.g. HELLO): ").upper()
yt_url = input("Enter YouTube video URL: ")

gesture_path = os.path.join(DATA_DIR, gesture_name)
os.makedirs(gesture_path, exist_ok=True)

video_path = os.path.join(VIDEOS_DIR, f"{gesture_name}.mp4")

# ========== DOWNLOAD ==========
if not os.path.exists(video_path):
    print(f"⬇️ Downloading {gesture_name}...")
    ydl_opts = {"outtmpl": video_path, "quiet": False, "format": "best"}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([yt_url])
else:
    print("✅ Video already downloaded.")

# ========== EXTRACT LANDMARKS ==========
cap = cv2.VideoCapture(video_path)
count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_hands.process(rgb)

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        pts = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark]).flatten()
        np.save(os.path.join(gesture_path, f"{count}.npy"), pts)
        count += 1

    cv2.imshow("Extracting landmarks", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"✅ Extracted {count} frames for '{gesture_name}' from video.")
