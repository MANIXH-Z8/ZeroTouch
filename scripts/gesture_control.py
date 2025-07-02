import cv2
import pyautogui
import numpy as np
import pandas as pd
import mediapipe as mp
import joblib
import time
from collections import deque

# --- Load model, scaler, and feature names ---
data = pd.read_csv(r"C:\Users\GENIUS\OneDrive\Desktop\Samayal daww\data\gesture_data.csv")
model = joblib.load("gesture_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = data.drop(columns='label').columns

# --- Kalman Filter ---
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

# --- Setup ---
kf = KalmanFilter()
screen_w, screen_h = pyautogui.size()
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.85, min_tracking_confidence=0.85)
mp_drawing = mp.solutions.drawing_utils

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

trail = deque(maxlen=10)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = cv2.GaussianBlur(rgb, (5, 5), 0)
    results = hands.process(rgb)
    frame_h, frame_w, _ = frame.shape
    current_time = time.time()
    mode_text = "Tracking"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Predict Gesture
            landmarks = [val for lm in hand_landmarks.landmark for val in (lm.x, lm.y, lm.z)]
            input_df = pd.DataFrame([landmarks], columns=feature_names)
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)[0]
            cv2.putText(frame, f'Gesture: {prediction}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Landmark points
            index_tip = hand_landmarks.landmark[8]
            middle_tip = hand_landmarks.landmark[12]
            thumb_tip = hand_landmarks.landmark[4]
            wrist = hand_landmarks.landmark[0]
            index = hand_landmarks.landmark[8]
            middle = hand_landmarks.landmark[12]
            ring = hand_landmarks.landmark[16]

            # Coordinates
            ix, iy = int(index_tip.x * frame_w), int(index_tip.y * frame_h)
            tx, ty = int(thumb_tip.x * frame_w), int(thumb_tip.y * frame_h)

            # Update trail
            trail.append((ix, iy))
            for i in range(1, len(trail)):
                if trail[i - 1] and trail[i]:
                    cv2.line(frame, trail[i - 1], trail[i], (0, 255, 0), 2)

            # Draw circles
            cv2.circle(frame, (ix, iy), 10, (0, 255, 255), 2)
            cv2.circle(frame, (tx, ty), 10, (255, 0, 255), 2)

            # Cursor movement
            cursor_x, cursor_y = int(index_tip.x * screen_w), int(index_tip.y * screen_h)
            smooth_x, smooth_y = kf.update((cursor_x, cursor_y))
            pyautogui.moveTo(smooth_x, smooth_y)

            # Pinch Click Detection
            pinch_dist = np.linalg.norm([index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y])
            if pinch_dist < PINCH_THRESHOLD:
                cv2.line(frame, (ix, iy), (tx, ty), (0, 255, 0), 2)
                cv2.circle(frame, ((ix + tx) // 2, (iy + ty) // 2), 30, (0, 255, 0), 3)
                cv2.putText(frame, "🖱️ Click Detected", (ix + 30, iy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                if not pinching:
                    pinch_start_time = current_time
                    pinching = True
                    clicked = False
                else:
                    if current_time - pinch_start_time >= HOLD_TIME and not clicked:
                        pyautogui.doubleClick()
                        clicked = True
                        last_click_time = current_time
            else:
                if pinching:
                    if not clicked and (current_time - last_click_time > CLICK_COOLDOWN):
                        if current_time - last_pinch_release_time <= DOUBLE_CLICK_GAP:
                            pyautogui.doubleClick()
                        else:
                            pyautogui.click()
                        last_click_time = current_time
                    last_pinch_release_time = current_time
                pinching = False
                clicked = False

            # Scroll Detection
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
                    last_scroll_time = current_time
                    mode_text = "Scrolling"
                elif below_thumb and scroll_speed < -10:
                    pyautogui.scroll(scroll_speed)
                    cv2.putText(frame, "⬇️ Scrolling Down", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                    last_scroll_time = current_time
                    mode_text = "Scrolling"

            # Drag Detection
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
                    dragging = True
                    mode_text = "Dragging"
                else:
                    pyautogui.mouseDown()
                    mode_text = "Dragging"
            elif dragging:
                pyautogui.mouseUp()
                dragging = False
    else:
        if dragging:
            pyautogui.mouseUp()
            dragging = False

    # --- Display Mode Status (Bottom Left) ---
    cv2.putText(frame, f"Mode: {mode_text}", (10, frame_h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 255, 200), 2)

    cv2.imshow("Gesture Mouse Control", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
