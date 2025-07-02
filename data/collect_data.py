import cv2
import csv
import os
import time
import mediapipe as mp

# Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Output CSV
DATA_PATH = r"C:\Users\GENIUS\OneDrive\Desktop\Samayal daww\data\gesture_data.csv"
LABEL = input("Enter gesture label (e.g., pinch, open, etc.): ")

# Create CSV if it doesn’t exist
if not os.path.exists(DATA_PATH):
    with open(DATA_PATH, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = ['label'] + [f"{i}_{axis}" for i in range(21) for axis in ['x', 'y', 'z']]
        writer.writerow(header)

cap = cv2.VideoCapture(0)
print("Press 's' to save sample, 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

        # Save if 's' is pressed
        if cv2.waitKey(1) & 0xFF == ord('s'):
            with open(DATA_PATH, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([LABEL] + landmarks)
            print(f"Sample saved for: {LABEL}")

    cv2.putText(frame, f"Gesture: {LABEL}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)
    cv2.imshow("Collecting Gesture Data", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
