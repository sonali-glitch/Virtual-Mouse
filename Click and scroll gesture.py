import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Start video capture
capture = cv2.VideoCapture(0)

# Get screen dimensions
screen_width, screen_height = pyautogui.size()

# To track time for double-click
last_click_time = 0

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
            
            # Extract finger landmarks
            index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            middle_finger = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

            # Get screen coordinates
            x, y = int(index_finger.x * width), int(index_finger.y * height)
            screen_x, screen_y = int(index_finger.x * screen_width), int(index_finger.y * screen_height)
            pyautogui.moveTo(screen_x, screen_y)

            # Left Click
            if abs(index_finger.y - thumb_tip.y) < 0.05:
                current_time = time.time()
                if current_time - last_click_time < 0.3:  # Detect double-click
                    pyautogui.doubleClick()
                    cv2.putText(frame, "Double Click", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    pyautogui.click()
                    cv2.putText(frame, "Left Click", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                last_click_time = current_time

            # Right Click
            if abs(middle_finger.y - thumb_tip.y) < 0.05:
                pyautogui.rightClick()
                cv2.putText(frame, "Right Click", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Scroll Function (Moving index & middle fingers together)
            if abs(index_finger.y - middle_finger.y) < 0.05:
                pyautogui.scroll(-10)
                cv2.putText(frame, "Scrolling", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow("Virtual Mouse", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
capture.release()
cv2.destroyAllWindows()
