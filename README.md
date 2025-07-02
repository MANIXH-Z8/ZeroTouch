# ✋ GestureMouse – Real-Time Air Gesture Mouse Control System

> A touchless hand gesture controller that turns your webcam into a fully functional air mouse — with gesture-based cursor movement, clicks, scroll, and drag.

---

## 🚀 Project Overview

**GestureMouse** is a real-time, AI-powered system that lets users control their computer using **hand gestures captured via webcam**. It combines computer vision and machine learning for smooth, accurate gesture recognition.

---

### 🔍 Core Features

| Feature                 | Description                                      |
| ----------------------- | ------------------------------------------------ |
| 🖱️ Cursor Control     | Index finger moves the mouse in real-time        |
| 🤏 Pinch Click          | Pinch gesture triggers single or double click    |
| ✋ Spread Palm Drag     | Five fingers + spread = drag and drop            |
| ⬆️⬇️ Air Scroll     | Two fingers above/below thumb = scroll up/down   |
| 🔄 Kalman Filter        | Smooth cursor motion with predictive filtering   |
| 🎯 Custom Gesture Model | Trained on personal data (`gesture_model.pkl`) |
| 💡 No External Hardware | Runs on any laptop with a webcam                 |

---

## 🧠 Tech Stack

| Technology    | Usage                     |
| ------------- | ------------------------- |
| Python        | Core programming language |
| OpenCV        | Video feed and drawing    |
| MediaPipe     | Hand landmark tracking    |
| PyAutoGUI     | Mouse event simulation    |
| Scikit-learn  | Gesture classification    |
| Kalman Filter | Smooth cursor movement    |

---

## 🗂️ Directory Structure


SAMAYAL DAWW/

│

├── data/

│   └── raw/

│       ├── gesture_data.csv         # Collected gesture points

│       └── collect_data.py

│

├── model/

│   ├── furnace.py                   # Training script

│   └── utils.py                     # Preprocessing utilities

│

├── real time control/

│   └── gesture_mouse.py             # Final working gesture control script

│

├── gesture_model.pkl                # Trained classifier

├── scaler.pkl                       # Scaler for gesture data

├── requirements.txt

├── README.md

**UI Features that will show during the execution :**

---
## ✅ Supported Gestures

| Gesture                         | Action                |
|----------------------------------|------------------------|
| Index Finger                    | Cursor movement        |
| Pinch (Thumb + Index)           | Single / Double Click  |
| Pinch + Hold                    | Double Click           |
| Index + Middle Above Thumb      | Scroll Up              |
| Index + Middle Below Thumb      | Scroll Down            |
| 5 Fingers Extended & Spread     | Start Drag             |
| Release Fingers                 | End Drag               |
---
## 🛠 How to Run

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```


### Step 2: Run the Application

python  "real time control/gesture_mouse.py"
