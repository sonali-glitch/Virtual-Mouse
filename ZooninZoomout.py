import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize Mediapipe Hands module
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Capture webcam feed
cap = cv2.VideoCapture(0)

# Variables to track pinch distance
previous_distance = None
zoom_threshold = 10  # Minimum distance change for zoom action
last_zoom_time = 0  # Prevent rapid zooming

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = img.shape
                lm_list.append((int(lm.x * w), int(lm.y * h)))

            # Get positions of thumb and index finger tips
            thumb_tip = lm_list[4]
            index_tip = lm_list[8]

            # Calculate Euclidean distance between thumb and index finger
            distance = ((thumb_tip[0] - index_tip[0]) ** 2 + (thumb_tip[1] - index_tip[1]) ** 2) ** 0.5

            # Time check to avoid rapid zooming
            current_time = time.time()
            if previous_distance is not None and current_time - last_zoom_time > 0.3:
                # Zoom In: If distance increases
                if distance - previous_distance > zoom_threshold:
                    pyautogui.hotkey('ctrl', '+')  # Zoom In
                    last_zoom_time = current_time
                    cv2.putText(img, "Zoom In", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Zoom Out: If distance decreases
                elif previous_distance - distance > zoom_threshold:
                    pyautogui.hotkey('ctrl', '-')  # Zoom Out
                    last_zoom_time = current_time
                    cv2.putText(img, "Zoom Out", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Update previous distance
            previous_distance = distance

            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Virtual Mouse - Zoom In & Out", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
