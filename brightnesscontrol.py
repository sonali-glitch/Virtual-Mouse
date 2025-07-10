import cv2
import mediapipe as mp
import numpy as np
import screen_brightness_control as sbc

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Open camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame and convert it to RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get coordinates of thumb and index finger
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Convert to pixel values
            h, w, _ = frame.shape
            x1, y1 = int(thumb_tip.x * w), int(thumb_tip.y * h)
            x2, y2 = int(index_tip.x * w), int(index_tip.y * h)

            # Calculate distance between thumb and index finger
            distance = np.hypot(x2 - x1, y2 - y1)

            # Map distance to brightness range (0 - 100)
            brightness = np.interp(distance, [20, 200], [0, 100])
            brightness = int(np.clip(brightness, 0, 100))

            # Set brightness
            sbc.set_brightness(brightness)

            # Display brightness value
            cv2.putText(frame, f'Brightness: {brightness}%', (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Brightness Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
