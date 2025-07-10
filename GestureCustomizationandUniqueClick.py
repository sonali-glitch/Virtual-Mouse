import cv2
import mediapipe as mp
import pyautogui
import json
import time

# Initialize Mediapipe Hands module
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Capture webcam feed
cap = cv2.VideoCapture(0)

# Load or define custom gestures
gesture_mapping = {
    "middle_click": [False, False, True, True, True],  # Default: Three fingers up
    "scroll_up": [True, True, True, True, True],       # All fingers up
    "scroll_down": [False, False, False, False, False] # All fingers down (fist)
}

# Load user-defined gestures from a file
try:
    with open("custom_gestures.json", "r") as file:
        gesture_mapping.update(json.load(file))
except FileNotFoundError:
    print("No custom gestures found, using defaults.")

# Gesture tracking
last_action_time = 0

def detect_fingers_up(lm_list):
    """Detect which fingers are up."""
    fingers = []
    tips = [8, 12, 16, 20]  # Index, middle, ring, pinky tips
    for tip in tips:
        fingers.append(lm_list[tip][1] < lm_list[tip - 2][1])  # Tip higher than PIP joint
    thumb_up = lm_list[4][0] > lm_list[3][0]  # Thumb rule (varies per hand)
    fingers.insert(0, thumb_up)
    return fingers

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = img.shape
                lm_list.append((int(lm.x * w), int(lm.y * h)))

            # Detect fingers up
            fingers_up = detect_fingers_up(lm_list)

            # Time check to prevent rapid execution
            current_time = time.time()
            if current_time - last_action_time > 0.5:

                # Middle Click: Default Gesture (Three Fingers Up)
                if fingers_up == gesture_mapping["middle_click"]:
                    pyautogui.middleClick()
                    last_action_time = current_time
                    cv2.putText(img, "Middle Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Scroll Up
                elif fingers_up == gesture_mapping["scroll_up"]:
                    pyautogui.scroll(5)
                    last_action_time = current_time
                    cv2.putText(img, "Scroll Up", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Scroll Down
                elif fingers_up == gesture_mapping["scroll_down"]:
                    pyautogui.scroll(-5)
                    last_action_time = current_time
                    cv2.putText(img, "Scroll Down", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Virtual Mouse - Middle Click & Custom Gestures", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
