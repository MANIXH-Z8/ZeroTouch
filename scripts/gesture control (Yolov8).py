import cv2
import pyautogui
import numpy as np
import pandas as pd
import time
import joblib
import mediapipe as mp
from ultralytics import YOLO
import os
import sys

# Detect if bundled with PyInstaller
if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS
else:
    base_path = os.path.dirname(__file__)

# Load CSV data and model/scaler files
data = pd.read_csv(os.path.join(base_path, 'gesture_data.csv'))
model = joblib.load(os.path.join(base_path, 'gesture_model.pkl'))
scaler = joblib.load(os.path.join(base_path, 'scaler.pkl'))

feature_names = data.drop(columns='label').columns

# --- Kalman Filter for Smoothing ---
class KalmanFilter:  
    def __init__(self):
        self.state = np.array([[0], [0], [0], [0]], dtype=np.float32)
        self.P = np.eye(4, dtype=np.float32)
        self.F = np.array([[1, 0, 1, 0],
                           [0, 1, 0, 1],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=np.float32)
        self.R = np.array([[50, 0],
                           [0, 50]], dtype=np.float32)
        self.Q = np.eye(4, dtype=np.float32) * 0.01

    def update(self, measurement):
        z = np.array([[measurement[0]], [measurement[1]]], dtype=np.float32)
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        y = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return int(self.state[0][0]), int(self.state[1][0])

# --- Load YOLOv8 ---
yolo_model = YOLO("yolov8n.pt")  # Replace with your custom trained YOLOv8 hand model

# --- MediaPipe Setup ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.85, min_tracking_confidence=0.85)
mp_drawing = mp.solutions.drawing_utils

# --- Initialize ---
kf = KalmanFilter()
screen_w, screen_h = pyautogui.size()
cap = cv2.VideoCapture(0)

# --- State Variables ---
PINCH_THRESHOLD = 0.045
HOLD_TIME = 1.0
DOUBLE_CLICK_GAP = 0.4
CLICK_COOLDOWN = 0.3
SCROLL_COOLDOWN = 0.1

pinching = False
clicked = False
dragging = False
last_click_time = 0
pinch_start_time = 0
last_pinch_release_time = 0
last_scroll_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = cv2.GaussianBlur(rgb, (5, 5), 0)
    frame_h, frame_w, _ = frame.shape
    current_time = time.time()
    hands_detected = []

    # --- YOLOv8 Detection ---
    yolo_results = yolo_model(rgb)[0]
    for det in yolo_results.boxes.data:
        x1, y1, x2, y2, conf, cls = det.tolist()
        if conf < 0.5:
            continue

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        hand_crop = rgb[y1:y2, x1:x2]
        if hand_crop.size == 0:
            continue

        hand_crop = cv2.resize(hand_crop, (frame.shape[1], frame.shape[0]))
        results = hands.process(hand_crop)

        if results.multi_hand_landmarks:
            for landmark in results.multi_hand_landmarks:
                hands_detected.append((landmark, x1, y1, x2, y2))

    # --- MediaPipe Landmark Analysis ---
    for hand_landmarks, x1, y1, x2, y2 in hands_detected:
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Gesture Classification
        landmarks = [val for lm in hand_landmarks.landmark for val in (lm.x, lm.y, lm.z)]
        input_df = pd.DataFrame([landmarks], columns=feature_names)
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        cv2.putText(frame, f'Gesture: {prediction}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Landmarks for control
        index_tip = hand_landmarks.landmark[8]
        middle_tip = hand_landmarks.landmark[12]
        thumb_tip = hand_landmarks.landmark[4]
        wrist = hand_landmarks.landmark[0]
        index = hand_landmarks.landmark[8]
        middle = hand_landmarks.landmark[12]
        ring = hand_landmarks.landmark[16]

        ix, iy = int(index_tip.x * frame_w), int(index_tip.y * frame_h)
        tx, ty = int(thumb_tip.x * frame_w), int(thumb_tip.y * frame_h)
        cv2.circle(frame, (ix, iy), 10, (0, 255, 255), 2)
        cv2.circle(frame, (tx, ty), 10, (255, 0, 255), 2)

        cursor_x, cursor_y = int(index_tip.x * screen_w), int(index_tip.y * screen_h)
        smooth_x, smooth_y = kf.update((cursor_x, cursor_y))
        pyautogui.moveTo(smooth_x, smooth_y)

        # --- Click Detection ---
        pinch_dist = np.linalg.norm([index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y])
        if pinch_dist < PINCH_THRESHOLD:
            cv2.line(frame, (ix, iy), (tx, ty), (0, 255, 0), 2)
            if not pinching:
                pinch_start_time = current_time
                pinching = True
                clicked = False
            else:
                if current_time - pinch_start_time >= HOLD_TIME and not clicked:
                    pyautogui.doubleClick()
                    print("🖱️ Double Click (held pinch)")
                    clicked = True
                    last_click_time = current_time
        else:
            if pinching:
                if not clicked and (current_time - last_click_time > CLICK_COOLDOWN):
                    if current_time - last_pinch_release_time <= DOUBLE_CLICK_GAP:
                        pyautogui.doubleClick()
                        print("🖱️ Double Click (quick pinch)")
                    else:
                        pyautogui.click()
                        print("🖱️ Single Click")
                    last_click_time = current_time
                last_pinch_release_time = current_time
            pinching = False
            clicked = False

        # --- Scroll Detection ---
        if (not pinching and not dragging and
            current_time - last_scroll_time > SCROLL_COOLDOWN):

            index_y = index_tip.y
            middle_y = middle_tip.y
            thumb_y = thumb_tip.y
            gap_threshold = 0.05

            above_thumb = index_y < thumb_y - gap_threshold and middle_y < thumb_y - gap_threshold
            below_thumb = index_y > thumb_y + gap_threshold and middle_y > thumb_y + gap_threshold

            avg_dist = ((thumb_y - index_y) + (thumb_y - middle_y)) / 2
            scroll_speed = int(avg_dist * 3000)

            if above_thumb and scroll_speed > 10:
                pyautogui.scroll(scroll_speed)
                cv2.putText(frame, "⬆️ Scrolling Up", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                print(f"⬆️ Scroll Up: {scroll_speed}")
                last_scroll_time = current_time

            elif below_thumb and scroll_speed < -10:
                pyautogui.scroll(scroll_speed)
                cv2.putText(frame, "⬇️ Scrolling Down", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                print(f"⬇️ Scroll Down: {scroll_speed}")
                last_scroll_time = current_time

        # --- Drag Detection ---
        finger_ids = [4, 8, 12, 16, 20]
        stretched_fingers = 0
        for i in finger_ids:
            dist = np.linalg.norm([
                hand_landmarks.landmark[i].x - wrist.x,
                hand_landmarks.landmark[i].y - wrist.y
            ])
            if dist > 0.25:
                stretched_fingers += 1

        spread_ok = (abs(index.x - middle.x) > 0.07 and abs(middle.x - ring.x) > 0.07)

        if stretched_fingers == 5 and spread_ok:
            if not dragging:
                pyautogui.mouseDown()
                print("🎯 Drag Start")
                dragging = True
            else:
                pyautogui.mouseDown()
                print("🎯 Dragging")
        elif dragging:
            pyautogui.mouseUp()
            print("🛑 Drag Stop")
            dragging = False

    # If hand lost
    if not hands_detected and dragging:
        pyautogui.mouseUp()
        print("🛑 Drag Stop (hand lost)")
        dragging = False

    cv2.imshow("Gesture Mouse Control", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
