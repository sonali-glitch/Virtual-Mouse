import cv2
import mediapipe as mp
import pyautogui
import time
import os
import platform

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Capture webcam feed
cap = cv2.VideoCapture(0)

last_action_time = 0  # Cooldown timer for actions

def detect_hand_gestures(lm_list):
    """ Detect gestures for window snapping and virtual assistant. """
    global last_action_time

    if len(lm_list) < 21:
        return

    # Get X coordinates for both hands
    left_x = lm_list[5][1]  # Wrist
    right_x = lm_list[17][1]  # Other wrist

    # Detect "Move Hands Apart" → Maximize Window
    if abs(left_x - right_x) > 250:  # Hands moved apart
        if time.time() - last_action_time > 2:
            print("🖥 Maximizing Window")
            pyautogui.hotkey("win", "up")
            last_action_time = time.time()

    # Detect "Move Hands Together" → Split Screen
    elif abs(left_x - right_x) < 100:  # Hands moved close
        if time.time() - last_action_time > 2:
            print("🖥 Splitting Screen")
            pyautogui.hotkey("win", "left")
            time.sleep(1)
            pyautogui.hotkey("win", "right")
            last_action_time = time.time()

    # Detect "Two Thumbs Up" → Open Virtual Assistant
    thumb_tips = [lm_list[4][1], lm_list[20][1]]  # Thumb tips
    index_knuckles = [lm_list[3][1], lm_list[19][1]]  # Index knuckles

    if thumb_tips[0] > index_knuckles[0] and thumb_tips[1] > index_knuckles[1]:  
        if time.time() - last_action_time > 3:
            print("🗣 Activating Virtual Assistant...")
            if platform.system() == "Windows":
                os.system("start ms-cortana://")
            elif platform.system() == "Darwin":  # Mac
                os.system("open -a Siri")
            else:  # Linux (Assuming Google Assistant)
                os.system("google-assistant")
            last_action_time = time.time()

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm_list = [(id, int(lm.x * img.shape[1]), int(lm.y * img.shape[0])) for id, lm in enumerate(hand_landmarks.landmark)]
            detect_hand_gestures(lm_list)

            # Draw hand landmarks
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Window Snap & Virtual Assistant Gesture", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
