import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Initialize Mediapipe Hand Detection
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Get screen size
screen_width, screen_height = pyautogui.size()

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Flip the frame horizontally for a mirrored effect
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape

    # Convert the frame to RGB for Mediapipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Get the tip of the index finger (Landmark 8)
            index_finger_tip = hand_landmarks.landmark[8]

            # Convert hand coordinates to screen coordinates
            cursor_x = int(index_finger_tip.x * screen_width)
            cursor_y = int(index_finger_tip.y * screen_height)

            # Move cursor
            pyautogui.moveTo(cursor_x, cursor_y, duration=0.1)

            # Draw hand landmarks on the frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the output frame
    cv2.imshow("Hand Gesture Virtual Mouse", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
