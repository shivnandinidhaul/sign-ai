import cv2
import mediapipe as mp
import joblib
import os

# ----------------------------
# Load model
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.pkl")

model = joblib.load(model_path)
print("Model loaded")

# ----------------------------
# MediaPipe setup
# ----------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

label = "No hand"
last_10_predictions = []

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb)

        if results.multi_hand_landmarks:

            hand = results.multi_hand_landmarks[0]

            # Draw landmarks
            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

            # Extract features
            row = []

            for lm in hand.landmark:
                row.extend([lm.x, lm.y, lm.z])

            # Predict
            prediction = model.predict([row])[0]

            # Smoothing
            last_10_predictions.append(prediction)

            if len(last_10_predictions) > 10:
                last_10_predictions.pop(0)

            label = max(
                set(last_10_predictions),
                key=last_10_predictions.count
            )

        else:
            label = "No hand"

        # Show prediction
        cv2.putText(
            frame,
            f"Predicted: {label}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("Sign AI", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()