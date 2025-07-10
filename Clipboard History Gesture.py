import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Start Webcam
cap = cv2.VideoCapture(0)

def open_clipboard_history():
    """Opens Windows Clipboard History using Win + V shortcut."""
    pyautogui.hotkey("win", "v")
    print("📋 Clipboard History Opened!")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the image for better tracking and convert to RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with MediaPipe
    results = hands.process(rgb_frame)

    clipboard_gesture_detected = False

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get landmark positions
            index_tip = hand_landmarks.landmark[8]
            middle_tip = hand_landmarks.landmark[12]
            ring_tip = hand_landmarks.landmark[16]

            index_base = hand_landmarks.landmark[5]
            middle_base = hand_landmarks.landmark[9]
            ring_base = hand_landmarks.landmark[13]

            # Check if three fingers are raised and apart
            if (index_tip.y < index_base.y and 
                middle_tip.y < middle_base.y and 
                ring_tip.y < ring_base.y and 
                abs(index_tip.x - middle_tip.x) > 0.05 and 
                abs(middle_tip.x - ring_tip.x) > 0.05):
                
                clipboard_gesture_detected = True

    # Open Clipboard History when the gesture is detected
    if clipboard_gesture_detected:
        open_clipboard_history()
        time.sleep(1)  # Prevent multiple triggers

    # Show Webcam Output
    cv2.putText(frame, "Show 3 Fingers Apart for Clipboard", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("Clipboard History Gesture", frame)

    # Exit on pressing 'Q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release Resources
cap.release()
cv2.destroyAllWindows()
