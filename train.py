import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# ----------------------------
# FIX: Always resolve absolute path
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(BASE_DIR, "data")

print("Looking for data folder at:", data_folder)

if not os.path.exists(data_folder):
    raise FileNotFoundError(f"Data folder not found at {data_folder}")

# ----------------------------
# Load dataset
# ----------------------------
X = []
y = []

for file in os.listdir(data_folder):
    if file.endswith(".csv"):
        label = file.replace(".csv", "")

        file_path = os.path.join(data_folder, file)

        df = pd.read_csv(file_path, header=None)

        for _, row in df.iterrows():
            X.append(row.values.tolist())
            y.append(label)

print(f"Loaded {len(X)} samples across {len(set(y))} classes")

# ----------------------------
# Train/test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ----------------------------
# Model
# ----------------------------
model = RandomForestClassifier(
    n_estimators=150,
    random_state=42
)

model.fit(X_train, y_train)

# ----------------------------
# Evaluation
# ----------------------------
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

print("Accuracy:", acc)

# ----------------------------
# Save model
# ----------------------------
model_path = os.path.join(BASE_DIR, "model.pkl")
joblib.dump(model, model_path)

print("Model saved at:", model_path)