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

def is_victory_sign(landmarks):
    """
    Detects a "V" shape (Victory sign) for Copy.
    - Index and Middle fingers extended
    - Ring and Pinky fingers folded
    """
    index_extended = landmarks[8][2] < landmarks[6][2]
    middle_extended = landmarks[12][2] < landmarks[10][2]
    ring_folded = landmarks[16][2] > landmarks[14][2]
    pinky_folded = landmarks[20][2] > landmarks[18][2]

    return index_extended and middle_extended and ring_folded and pinky_folded

def is_p_shape(landmarks):
    """
    Detects a "P" shape gesture for Paste.
    - Index finger extended
    - Thumb extended
    - Other fingers folded
    """
    index_extended = landmarks[8][2] < landmarks[6][2]
    thumb_extended = landmarks[4][1] < landmarks[3][1]  # Thumb pointing left/right
    middle_folded = landmarks[12][2] > landmarks[10][2]
    ring_folded = landmarks[16][2] > landmarks[14][2]
    pinky_folded = landmarks[20][2] > landmarks[18][2]

    return index_extended and thumb_extended and middle_folded and ring_folded and pinky_folded

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm_list = [(id, int(lm.x * img.shape[1]), int(lm.y * img.shape[0])) for id, lm in enumerate(hand_landmarks.landmark)]
            
            # Detect gestures
            if time.time() - last_action_time > 1:  # Prevent multiple triggers
                if is_victory_sign(lm_list):
                    print("📋 Copy Gesture Detected (Ctrl + C)")
                    pyautogui.hotkey('ctrl', 'c')
                    last_action_time = time.time()

                elif is_p_shape(lm_list):
                    print("📋 Paste Gesture Detected (Ctrl + V)")
                    pyautogui.hotkey('ctrl', 'v')
                    last_action_time = time.time()

            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Gesture-Based Copy/Paste", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

