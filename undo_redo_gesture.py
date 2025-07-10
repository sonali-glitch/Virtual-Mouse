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

def detect_swipe_direction(landmarks):
    """
    Detects if two fingers are swiping left (Undo) or right (Redo).
    - Index & Middle finger should be extended.
    - If they move significantly left -> Undo
    - If they move significantly right -> Redo
    """
    index_tip = np.array([landmarks[8][1], landmarks[8][2]])  # Index finger tip
    middle_tip = np.array([landmarks[12][1], landmarks[12][2]])  # Middle finger tip

    # Calculate the average position of both fingers
    avg_x = (index_tip[0] + middle_tip[0]) / 2  # X-coordinate of center of two fingers

    # Track the movement direction
    global prev_x
    if prev_x is not None:
        movement = avg_x - prev_x  # Difference in X movement

        if movement < -50:  # Move left (Undo)
            print("↩️ Undo Gesture Detected! (Ctrl + Z)")
            pyautogui.hotkey('ctrl', 'z')
            time.sleep(0.5)  # Prevent multiple triggers

        elif movement > 50:  # Move right (Redo)
            print("↪️ Redo Gesture Detected! (Ctrl + Y)")
            pyautogui.hotkey('ctrl', 'y')
            time.sleep(0.5)  # Prevent multiple triggers

    prev_x = avg_x  # Update previous position

# Initialize previous X position
prev_x = None  

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm_list = [(id, int(lm.x * img.shape[1]), int(lm.y * img.shape[0])) for id, lm in enumerate(hand_landmarks.landmark)]

            detect_swipe_direction(lm_list)  # Check for Undo/Redo gestures

            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Undo/Redo Gesture", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
