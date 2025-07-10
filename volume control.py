import cv2
import mediapipe as mp
import numpy as np
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
import math

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initialize camera
cap = cv2.VideoCapture(0)

# Initialize audio control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get landmark positions
            thumb_tip = hand_landmarks.landmark[4]
            index_finger_tip = hand_landmarks.landmark[8]

            # Convert coordinates
            h, w, _ = frame.shape
            x1, y1 = int(thumb_tip.x * w), int(thumb_tip.y * h)
            x2, y2 = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

            # Draw line between thumb and index finger
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (x1, y1), 5, (255, 0, 0), -1)
            cv2.circle(frame, (x2, y2), 5, (255, 0, 0), -1)

            # Calculate distance
            length = math.hypot(x2 - x1, y2 - y1)

            # Map distance to volume range (0 to 100)
            min_vol, max_vol = -65, 0  # System volume range
            volume_level = np.interp(length, [20, 150], [min_vol, max_vol])
            volume.SetMasterVolumeLevel(volume_level, None)

            # Display volume level
            vol_percentage = np.interp(volume_level, [min_vol, max_vol], [0, 100])
            cv2.putText(frame, f"Volume: {int(vol_percentage)}%", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show camera feed
    cv2.imshow("Volume Control", frame)

    # Exit on 'Q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
