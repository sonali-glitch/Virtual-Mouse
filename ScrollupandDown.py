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

# Time tracking to control scrolling speed
last_scroll_time = 0

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

            # Tip positions of all fingers
            thumb_tip = lm_list[4]
            index_tip = lm_list[8]
            middle_tip = lm_list[12]
            ring_tip = lm_list[16]
            pinky_tip = lm_list[20]

            # Condition for scroll up: All fingers up (open palm)
            if (index_tip[1] < lm_list[6][1] and
                middle_tip[1] < lm_list[10][1] and
                ring_tip[1] < lm_list[14][1] and
                pinky_tip[1] < lm_list[18][1]):

                current_time = time.time()
                if current_time - last_scroll_time > 0.2:
                    pyautogui.scroll(5)  # Scroll up
                    last_scroll_time = current_time  # Update last scroll time
                    cv2.putText(img, "Scroll Up", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Condition for scroll down: All fingers down (closed fist)
            elif (index_tip[1] > lm_list[6][1] and
                  middle_tip[1] > lm_list[10][1] and
                  ring_tip[1] > lm_list[14][1] and
                  pinky_tip[1] > lm_list[18][1]):

                current_time = time.time()
                if current_time - last_scroll_time > 0.2:
                    pyautogui.scroll(-5)  # Scroll down
                    last_scroll_time = current_time  # Update last scroll time
                    cv2.putText(img, "Scroll Down", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Virtual Mouse - Scroll Up & Down", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
