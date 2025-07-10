import cv2
import mediapipe as mp
import pyautogui

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Capture webcam feed
cap = cv2.VideoCapture(0)

previous_angle = None  # Track previous hand rotation
language_switched = False  # Prevent multiple switches on small rotations

def switch_keyboard_language():
    """Simulates keyboard shortcut to switch input language (Windows & macOS)."""
    print("🌍 Switching Keyboard Language...")
    pyautogui.hotkey("alt", "shift")  # Windows shortcut
    # pyautogui.hotkey("command", "space")  # macOS shortcut (Uncomment for macOS)

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get key wrist points
            wrist_x = hand_landmarks.landmark[0].x
            wrist_y = hand_landmarks.landmark[0].y
            index_finger_x = hand_landmarks.landmark[5].x
            index_finger_y = hand_landmarks.landmark[5].y

            # Calculate hand rotation angle
            angle = (index_finger_y - wrist_y) / (index_finger_x - wrist_x + 0.0001)  # Avoid division by zero

            # Detect significant hand rotation
            if previous_angle is not None:
                rotation_change = abs(angle - previous_angle)
                if rotation_change > 1.5 and not language_switched:
                    switch_keyboard_language()
                    language_switched = True  # Prevent multiple quick switches

            previous_angle = angle  # Update previous angle

    else:
        language_switched = False  # Reset when hand is not detected

    cv2.imshow("Multi-Language Gesture", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
