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

def is_fist_closed(landmarks):
    """Check if all fingers are folded (closed fist)"""
    finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky tips
    finger_base = [6, 10, 14, 18]  # Finger base points

    for tip, base in zip(finger_tips, finger_base):
        if landmarks[tip][1] < landmarks[base][1]:  # If any finger tip is above the base, hand is open
            return False
    return True

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
                lm_list.append((id, int(lm.x * w), int(lm.y * h)))

            if is_fist_closed(lm_list):
                print("Screen Lock Gesture Detected! Locking Screen...")
                pyautogui.hotkey('win', 'l')  # Locks the screen
                time.sleep(2)  # Prevents multiple triggers

            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Screen Lock Gesture", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
