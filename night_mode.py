import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize Mediapipe Hands module
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Capture webcam feed
cap = cv2.VideoCapture(0)

last_action_time = time.time()  # Prevent multiple triggers
night_mode_active = False  # Track night mode status

def is_fist(landmarks):
    """
    Detects if a hand is in a closed fist position.
    - Checks if fingertips are below their respective knuckle points.
    """
    index_finger = landmarks[8][2] > landmarks[5][2]
    middle_finger = landmarks[12][2] > landmarks[9][2]
    ring_finger = landmarks[16][2] > landmarks[13][2]
    pinky_finger = landmarks[20][2] > landmarks[17][2]

    return index_finger and middle_finger and ring_finger and pinky_finger

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    num_fists = 0  # Count number of fists detected

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm_list = [(id, int(lm.x * img.shape[1]), int(lm.y * img.shape[0])) for id, lm in enumerate(hand_landmarks.landmark)]
            
            if is_fist(lm_list):
                num_fists += 1  # Increase fist count

            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # If both hands are fists, toggle night mode
    if num_fists == 2 and time.time() - last_action_time > 1:
        night_mode_active = not night_mode_active  # Toggle night mode

        if night_mode_active:
            print("🌙 Night Mode Activated!")
            pyautogui.hotkey('win', 'ctrl', 'c')  # Windows built-in dark mode toggle
        else:
            print("☀️ Night Mode Deactivated!")
            pyautogui.hotkey('win', 'ctrl', 'c')

        last_action_time = time.time()  # Prevent multiple triggers

    cv2.imshow("Night Mode Gesture", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
