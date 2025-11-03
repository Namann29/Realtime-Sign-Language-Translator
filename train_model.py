import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
import joblib

DATA_PATH = "data"
gestures = os.listdir(DATA_PATH)

X, y = [], []

for i, gesture in enumerate(gestures):
    files = os.listdir(os.path.join(DATA_PATH, gesture))
    for f in files:
        data = np.load(os.path.join(DATA_PATH, gesture, f))
        X.append(data)
        y.append(i)

X, y = np.array(X), np.array(y)

model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

joblib.dump((model, gestures), "sign_model.pkl")
print("✅ Model trained and saved!")
