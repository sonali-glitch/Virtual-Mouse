import cv2
import mediapipe as mp
import pyautogui

# Initialize Mediapipe Hands module
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Capture webcam feed
cap = cv2.VideoCapture(0)

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

            # Index finger tip and middle finger tip positions
            index_tip = lm_list[8]
            middle_tip = lm_list[12]

            # Condition for left click: Index finger up, middle finger down
            if index_tip[1] < lm_list[6][1] and middle_tip[1] > lm_list[10][1]:
                pyautogui.click()
                cv2.putText(img, "Left Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Virtual Mouse", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
