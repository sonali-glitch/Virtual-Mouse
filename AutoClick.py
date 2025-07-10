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

# Auto Click Variables
hover_start_time = None
hover_threshold = 1  # Time in seconds before auto-click
last_position = None

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

            # Get the position of the index finger tip
            index_tip = lm_list[8]

            # Move the mouse cursor
            pyautogui.moveTo(index_tip[0], index_tip[1])

            # Auto-click logic
            if last_position and abs(index_tip[0] - last_position[0]) < 5 and abs(index_tip[1] - last_position[1]) < 5:
                if hover_start_time is None:
                    hover_start_time = time.time()
                elif time.time() - hover_start_time > hover_threshold:
                    pyautogui.click()
                    hover_start_time = None  # Reset timer after clicking
                    cv2.putText(img, "Auto Click!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                hover_start_time = None  # Reset if movement is detected

            last_position = index_tip  # Store last finger position

            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Virtual Mouse - Auto Click", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
