import cv2
import mediapipe as mp
import csv
import os
print("starting program")
print("asking for sign label")
label = input("Enter sign label: ").strip().upper()

print("LABEL IS:", label)

# Create data folder if it doesn't exist
os.makedirs("data", exist_ok=True)

# CSV file for this sign
csv_file = f"data/{label}.csv"

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)

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

        # Draw hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

        cv2.imshow("Sign AI", frame)

        key = cv2.waitKey(1) & 0xFF

        # Debug keyboard input
        if key != 255:
            print("KEY:", key)

        # Save sample when S is pressed
        if key == ord("s"):
            print("S pressed!")
            print(f"Saved sample for {label}")

            if results.multi_hand_landmarks:
                print("Hand detected!")

                hand_landmarks = results.multi_hand_landmarks[0]

                row = []

                for lm in hand_landmarks.landmark:
                    row.extend([lm.x, lm.y, lm.z])

                with open(csv_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(row)

                print(f"Saved sample for {label}")

        # ESC to quit
        if key == 27:
            break

cap.release()
cv2.destroyAllWindows()