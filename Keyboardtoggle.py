import cv2
import mediapipe as mp
import pyautogui
import os
import platform
import time

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Capture webcam feed
cap = cv2.VideoCapture(0)

keyboard_open = False  # Track keyboard state
last_action_time = 0  # Cooldown timer for gestures

def toggle_keyboard():
    """ Toggle the on-screen keyboard based on the OS """
    global keyboard_open

    if platform.system() == "Windows":
        if keyboard_open:
            os.system("taskkill /IM TabTip.exe /F")  # Close On-Screen Keyboard
        else:
            os.system("start osk")  # Open On-Screen Keyboard
    elif platform.system() == "Darwin":  # macOS
        if keyboard_open:
            os.system("osascript -e 'tell application \"System Events\" to key code 53'")  # Close keyboard
        else:
            os.system("open -a Keyboard")  # Open Keyboard
    elif platform.system() == "Linux":
        if keyboard_open:
            os.system("killall onboard")  # Close On-Screen Keyboard
        else:
            os.system("onboard")  # Open On-Screen Keyboard

    keyboard_open = not keyboard_open

def detect_t_shape(lm_list):
    """ Detect 'T' shape with index + middle fingers across the palm """
    global last_action_time

    if len(lm_list) < 21:
        return False

    index_tip = lm_list[8]
    middle_tip = lm_list[12]
    palm_base = lm_list[0]

    # Condition: Index & Middle fingers are extended horizontally over the palm
    if abs(index_tip[1] - middle_tip[1]) < 20 and index_tip[2] < palm_base[2] and middle_tip[2] < palm_base[2]:
        return True
    return False

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm_list = [(id, int(lm.x * img.shape[1]), int(lm.y * img.shape[0])) for id, lm in enumerate(hand_landmarks.landmark)]

            if detect_t_shape(lm_list):
                if time.time() - last_action_time > 2:  # Prevent multiple triggers
                    print("⌨️ Toggling On-Screen Keyboard...")
                    toggle_keyboard()
                    last_action_time = time.time()

            # Draw hand landmarks
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Keyboard Toggle Gesture", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
