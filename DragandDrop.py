import cv2
import mediapipe as mp
import pyautogui

# Initialize Mediapipe Hands module
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Capture webcam feed
cap = cv2.VideoCapture(0)

# Dragging state
dragging = False

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

            # Tip and base positions of thumb and index finger
            thumb_tip = lm_list[4]
            index_tip = lm_list[8]

            # Compute distance between thumb tip and index finger tip
            distance = ((thumb_tip[0] - index_tip[0]) ** 2 + (thumb_tip[1] - index_tip[1]) ** 2) ** 0.5

            # Move the mouse to the index finger's position
            pyautogui.moveTo(index_tip[0], index_tip[1])

            # If distance is small, initiate drag
            if distance < 40 and not dragging:
                pyautogui.mouseDown()
                dragging = True
                cv2.putText(img, "Dragging", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # If distance is large, release the drag
            elif distance > 50 and dragging:
                pyautogui.mouseUp()
                dragging = False
                cv2.putText(img, "Dropped", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Virtual Mouse - Drag & Drop", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
