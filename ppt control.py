import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Open webcam
cap = cv2.VideoCapture(0)

prev_x = None  # Store previous x-coordinate of hand
slide_triggered = False
start_time = None  # Track time for slideshow start gesture

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame for mirror effect
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with MediaPipe
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get index finger position
            index_finger = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x, y = int(index_finger.x * frame.shape[1]), int(index_finger.y * frame.shape[0])

            # Detect left or right swipe for slide control
            if prev_x is not None:
                if x - prev_x > 100:  # Swipe Right -> Next Slide
                    pyautogui.press("right")
                    cv2.putText(frame, "Next Slide", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                elif x - prev_x < -100:  # Swipe Left -> Previous Slide
                    pyautogui.press("left")
                    cv2.putText(frame, "Previous Slide", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            prev_x = x  # Update previous x-coordinate

            # Detect palm raise for starting/stopping slideshow
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            if wrist.y < index_finger.y - 0.1:  # Palm is raised
                if start_time is None:
                    start_time = time.time()
                elif time.time() - start_time > 2:  # Hold for 2 seconds
                    pyautogui.press("f5")  # Start slideshow
                    cv2.putText(frame, "Slideshow Started", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                    start_time = None  # Reset timer
            else:
                start_time = None  # Reset if hand is lowered

    # Show frame
    cv2.imshow("PPT Gesture Control", frame)

    # Press 'Q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
