import cv2
import numpy as np
import mediapipe as mp

# Initialize Mediapipe Hand Detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Initialize the Canvas
canvas = np.zeros((480, 640, 3), dtype=np.uint8)

# Open webcam
cap = cv2.VideoCapture(0)
prev_x, prev_y = 0, 0
color = (255, 0, 0)  # Default color is blue

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)  # Flip the frame horizontally
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            index_finger = hand_landmarks.landmark[8]  # Index finger tip
            h, w, _ = frame.shape
            x, y = int(index_finger.x * w), int(index_finger.y * h)
            
            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = x, y
            
            cv2.line(canvas, (prev_x, prev_y), (x, y), color, 5)
            prev_x, prev_y = x, y
    
    frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
    cv2.imshow("Hand Gesture Paint", frame)
    
    key = cv2.waitKey(1)
    if key == 27:  # Press ESC to exit
        break
    elif key == ord('c'):  # Press 'c' to clear the canvas
        canvas.fill(0)
        prev_x, prev_y = 0, 0
    elif key == ord('r'):  # Press 'r' for red
        color = (0, 0, 255)
    elif key == ord('g'):  # Press 'g' for green
        color = (0, 255, 0)
    elif key == ord('b'):  # Press 'b' for blue
        color = (255, 0, 0)
    elif key == ord('y'):  # Press 'y' for yellow
        color = (0, 255, 255)

cap.release()
cv2.destroyAllWindows()
