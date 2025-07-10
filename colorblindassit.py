import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import json
import time

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Capture webcam feed
cap = cv2.VideoCapture(0)

# Gesture Tracking
last_action_time = 0  
color_filter_enabled = False  # Track color filter mode
custom_gestures = {}  # Dictionary to store user-defined gestures

def apply_color_filter(frame):
    """ Apply a color filter to assist color blindness """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    frame[:, :, 1] = cv2.add(frame[:, :, 1], 50)  # Increase saturation for color enhancement
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def detect_three_finger_sideways(lm_list):
    """ Detect three fingers extended sideways for Color Blind Assist Mode """
    global last_action_time, color_filter_enabled

    if len(lm_list) < 21:
        return False

    index_tip, middle_tip, ring_tip = lm_list[8], lm_list[12], lm_list[16]  
    wrist = lm_list[0]  

    # Condition: Three fingers extended horizontally near the wrist level
    if abs(index_tip[2] - wrist[2]) < 50 and abs(middle_tip[2] - wrist[2]) < 50 and abs(ring_tip[2] - wrist[2]) < 50:
        return True
    return False

def detect_custom_gesture(lm_list):
    """ Detect user-defined gestures stored in 'custom_gestures.json' """
    global last_action_time

    try:
        with open("custom_gestures.json", "r") as file:
            custom_gestures = json.load(file)

        for gesture_name, coords in custom_gestures.items():
            stored_coords = np.array(coords)
            current_coords = np.array([lm[1:] for lm in lm_list])

            if np.allclose(stored_coords, current_coords, atol=30):  # Check similarity
                if time.time() - last_action_time > 2:
                    print(f"✅ Activating Custom Gesture: {gesture_name}")
                    pyautogui.hotkey(custom_gestures[gesture_name]["action"])
                    last_action_time = time.time()

    except FileNotFoundError:
        pass  # No custom gestures found

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm_list = [(id, int(lm.x * img.shape[1]), int(lm.y * img.shape[0])) for id, lm in enumerate(hand_landmarks.landmark)]

            # Check for Color Blind Assist Mode
            if detect_three_finger_sideways(lm_list):
                if time.time() - last_action_time > 2:
                    color_filter_enabled = not color_filter_enabled
                    print("🎨 Color Blind Assist Mode:", "ON" if color_filter_enabled else "OFF")
                    last_action_time = time.time()

            # Check for custom gestures
            detect_custom_gesture(lm_list)

            # Draw hand landmarks
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    if color_filter_enabled:
        img = apply_color_filter(img)

    cv2.imshow("Color Blind Assist & Custom Gesture Mapping", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
