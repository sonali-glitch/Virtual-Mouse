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

fist_count = 0  # Count consecutive fist detections
last_fist_time = 0  # Track time of last fist detection

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

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm_list = [(id, int(lm.x * img.shape[1]), int(lm.y * img.shape[0])) for id, lm in enumerate(hand_landmarks.landmark)]
            
            if is_fist(lm_list):
                current_time = time.time()
                
                # Detect two rapid fist gestures within 1.5 seconds
                if current_time - last_fist_time < 1.5:
                    fist_count += 1
                else:
                    fist_count = 1  # Reset count if time gap is too long

                last_fist_time = current_time  # Update last detected fist time

                # If two fists detected rapidly, lock the PC
                if fist_count == 2:
                    print("🔒 Emergency Lock Triggered!")
                    pyautogui.hotkey('win', 'l')
                    fist_count = 0  # Reset counter after action

            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Emergency Lock Gesture", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
