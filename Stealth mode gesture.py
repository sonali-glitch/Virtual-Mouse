import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Open webcam
cap = cv2.VideoCapture(0)

cover_start_time = None  # Store the time when the hand covers the camera
stealth_activated = False  # Flag to prevent repeated triggers

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip image for a mirror effect
    frame = cv2.flip(frame, 1)

    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Check if hand is detected
    if results.multi_hand_landmarks:
        cover_start_time = None  # Reset the timer if a hand is detected
        stealth_activated = False  # Allow activation again
    else:
        # No hand detected -> Check if screen is covered (black screen)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray_frame)  # Calculate average brightness

        if brightness < 20:  # If screen is mostly dark (hand fully covers the camera)
            if cover_start_time is None:
                cover_start_time = time.time()
            
            elapsed_time = time.time() - cover_start_time

            if elapsed_time > 2 and not stealth_activated:  # If covered for 2 seconds
                print("🕶️ Stealth Mode Activated: Hiding all windows!")
                pyautogui.hotkey("win", "d")  # Hide all windows
                stealth_activated = True  # Prevent repeated triggers
        else:
            cover_start_time = None  # Reset if the hand is moved away

    # Display instructions
    cv2.putText(frame, "Cover the camera for 2 seconds to ACTIVATE STEALTH MODE", 
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("Stealth Mode Gesture", frame)

    # Exit on 'Q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
