import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe and OpenCV
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands()
cap = cv2.VideoCapture(0)

# Create a black image for drawing
canvas = np.zeros((480, 640, 3), dtype=np.uint8)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and convert the frame to RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect hands
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get coordinates of the index finger tip
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            h, w, _ = frame.shape
            cx, cy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

            # Draw on the canvas
            cv2.circle(canvas, (cx, cy), 5, (255, 255, 255), -1)

    # Combine the frame and the canvas
    combined = cv2.add(frame, canvas)

    # Display the result
    cv2.imshow('Virtual Drawing', combined)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
