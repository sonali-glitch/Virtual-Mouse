import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

prev_y = None  # Store previous y-coordinate of hand
minimize_triggered = False
cooldown_time = 2  # Cooldown to prevent accidental triggers
last_minimize_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip image for a mirror effect
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with MediaPipe
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            current_y = wrist.y  # Y-coordinate of wrist

            # Detect rapid downward movement
            if prev_y is not None and (prev_y - current_y) > 0.15:  # Adjust sensitivity if needed
                if not minimize_triggered and (time.time() - last_minimize_time) > cooldown_time:
                    print("🔽 Minimizing All Windows...")
                    pyautogui.hotkey("win", "d")  # Win + D minimizes all windows
                    minimize_triggered = True
                    last_minimize_time = time.time()

            prev_y = current_y

    else:
        minimize_triggered = False  # Reset if no hand is detected

    # Display instructions
    cv2.putText(frame, "Move hand down rapidly to MINIMIZE", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("Quick Minimize Gesture", frame)

    # Exit on 'Q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
