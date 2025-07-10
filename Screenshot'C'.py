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

def is_c_shape(landmarks):
    """
    Detects if the hand is in a 'C' shape.
    Condition: Thumb and Index finger form a curve while other fingers are slightly folded.
    """
    thumb_tip = np.array([landmarks[4][1], landmarks[4][2]])  # Thumb tip
    index_tip = np.array([landmarks[8][1], landmarks[8][2]])  # Index finger tip
    index_base = np.array([landmarks[5][1], landmarks[5][2]])  # Index finger base

    # Check if thumb and index are forming a 'C' shape
    thumb_index_dist = np.linalg.norm(thumb_tip - index_tip)
    index_straight = np.linalg.norm(index_tip - index_base)

    return thumb_index_dist < 50 and index_straight > 50  # Adjust thresholds as needed

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm_list = [(id, int(lm.x * img.shape[1]), int(lm.y * img.shape[0])) for id, lm in enumerate(hand_landmarks.landmark)]

            if is_c_shape(lm_list):
                print("C Gesture Detected! Taking Screenshot...")
                screenshot = pyautogui.screenshot()
                screenshot.save("screenshot.png")
                print("Screenshot saved as screenshot.png")
                time.sleep(1)  # Prevent multiple triggers

            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Screenshot Gesture", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
