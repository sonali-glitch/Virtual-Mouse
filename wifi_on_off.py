import cv2
import mediapipe as mp
import subprocess

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Start Video Capture
cap = cv2.VideoCapture(0)

def toggle_wifi(state):
    """Enable or disable WiFi using system commands."""
    if state == "on":
        subprocess.run("netsh interface set interface name='Wi-Fi' admin=enabled", shell=True)
        print("WiFi Turned ON")
    elif state == "off":
        subprocess.run("netsh interface set interface name='Wi-Fi' admin=disabled", shell=True)
        print("WiFi Turned OFF")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get landmark positions
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y

            # Gesture: Thumb Up (Enable WiFi)
            if thumb_tip < index_tip:
                toggle_wifi("on")

            # Gesture: Thumb Down (Disable WiFi)
            if thumb_tip > index_tip:
                toggle_wifi("off")

    # Show Video Feed
    cv2.imshow("Hand Gesture WiFi Control", frame)

    # Exit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()