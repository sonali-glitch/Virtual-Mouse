import cv2
import mediapipe as mp
import time
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

shutdown_start_time = None  # Time when both hands are detected

def shutdown_system():
    """Shuts down the computer."""
    print("🛑 System shutting down...")
    os.system("shutdown /s /t 5")  # Windows shutdown command (5s timer)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the image for better tracking
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with MediaPipe
    results = hands.process(rgb_frame)
    
    raised_hands = 0  # Count number of hands raised

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            wrist = hand_landmarks.landmark[0]
            index_tip = hand_landmarks.landmark[8]
            middle_tip = hand_landmarks.landmark[12]
            ring_tip = hand_landmarks.landmark[16]
            pinky_tip = hand_landmarks.landmark[20]

            # Check if all fingers are above the wrist (hand is raised)
            if (index_tip.y < wrist.y and 
                middle_tip.y < wrist.y and 
                ring_tip.y < wrist.y and 
                pinky_tip.y < wrist.y):
                raised_hands += 1

    # Detect both hands raised for 5 seconds
    if raised_hands >= 2:
        if shutdown_start_time is None:
            shutdown_start_time = time.time()
        elif time.time() - shutdown_start_time >= 5:  # 5 seconds hold
            shutdown_system()
            break
    else:
        shutdown_start_time = None  # Reset timer if hands are lowered

    # Display status on screen
    cv2.putText(frame, "Raise both hands for 5s to SHUTDOWN", (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    
    cv2.imshow("Shutdown Gesture Detection", frame)

    # Exit on 'Q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
