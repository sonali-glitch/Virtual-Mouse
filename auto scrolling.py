import cv2
import mediapipe as mp
import pyautogui

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Start video capture
capture = cv2.VideoCapture(0)

# Get screen dimensions
screen_width, screen_height = pyautogui.size()

# Reference Y position for scrolling
prev_y = None

while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        break

    # Flip frame for a mirror effect
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)
    height, width, _ = frame.shape

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get index finger tip position
            index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            finger_y = int(index_finger.y * height)

            # Auto scroll logic
            if prev_y is not None:
                if finger_y < prev_y - 20:  # Hand moves up
                    pyautogui.scroll(10)  # Scroll up
                    cv2.putText(frame, "Scrolling Up", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                elif finger_y > prev_y + 20:  # Hand moves down
                    pyautogui.scroll(-10)  # Scroll down
                    cv2.putText(frame, "Scrolling Down", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            prev_y = finger_y  # Update previous position

    # Show the frame
    cv2.imshow("Auto Scroll", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
capture.release()
cv2.destroyAllWindows()
