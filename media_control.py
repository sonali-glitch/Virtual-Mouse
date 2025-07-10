import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time

# Initialize Mediapipe Hands module
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Capture webcam feed
cap = cv2.VideoCapture(0)

prev_x = None  # Initialize previous X position
last_action_time = time.time()  # Prevent multiple triggers

def detect_gesture(landmarks, img):
    """
    Detects media control gestures:
    - Open Palm -> Play/Pause (Spacebar)
    - Swipe Right -> Next Song (Ctrl + Right Arrow)
     -   Swipe Left -> Previous Song (Ctrl + Left Arrow)
    """
    global prev_x, last_action_time

    index_tip = np.array([landmarks[8][1], landmarks[8][2]])  # Index finger tip
    middle_tip = np.array([landmarks[12][1], landmarks[12][2]])  # Middle finger tip
    ring_tip = np.array([landmarks[16][1], landmarks[16][2]])  # Ring finger tip
    pinky_tip = np.array([landmarks[20][1], landmarks[20][2]])  # Pinky tip
    thumb_tip = np.array([landmarks[4][1], landmarks[4][2]])  # Thumb tip

    # Calculate distances to check if all fingers are extended
    palm_open = (
        index_tip[1] < landmarks[6][2] and  # Index finger extended
        middle_tip[1] < landmarks[10][2] and  # Middle finger extended
        ring_tip[1] < landmarks[14][2] and  # Ring finger extended
        pinky_tip[1] < landmarks[18][2]  # Pinky finger extended
    )

    # Prevent multiple triggers with a delay
    if time.time() - last_action_time < 0.5:
        return  

    if palm_open:  # Play/Pause
        print("▶️ Play/Pause Gesture Detected (Spacebar)")
        pyautogui.press('space')  # Simulate spacebar
        last_action_time = time.time()
        return

    if prev_x is not None:
        movement = index_tip[0] - prev_x  # Difference in X movement

        if movement > 50:  # Swipe Right (Next Song)
            print("⏩ Next Song Gesture Detected (Ctrl + Right Arrow)")
            pyautogui.hotkey('ctrl', 'right')
            last_action_time = time.time()

        elif movement < -50:  # Swipe Left (Previous Song)
            print("⏪ Previous Song Gesture Detected (Ctrl + Left Arrow)")
            pyautogui.hotkey('ctrl', 'left')
            last_action_time = time.time()

    prev_x = index_tip[0]  # Update previous position

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm_list = [(id, int(lm.x * img.shape[1]), int(lm.y * img.shape[0])) for id, lm in enumerate(hand_landmarks.landmark)]

            detect_gesture(lm_list, img)  # Check for media control gestures

            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Media Control Gestures", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
