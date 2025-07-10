import cv2
import mediapipe as mp
import pyautogui
import platform
import os

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Capture webcam feed
cap = cv2.VideoCapture(0)

def open_voice_assistant():
    """Opens Google Assistant, Siri, or Cortana based on OS."""
    os_name = platform.system()

    if os_name == "Windows":
        print("🎙️ Opening Cortana...")
        pyautogui.hotkey("win", "c")  # Windows Cortana shortcut
    elif os_name == "Darwin":  # macOS
        print("🎙️ Opening Siri...")
        os.system("open -a Siri")
    elif os_name == "Linux":
        print("🎙️ Opening Google Assistant...")
        os.system("google-assistant-sdk")  # Requires Google Assistant SDK installed

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    thumbs_up_count = 0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            thumb_tip = hand_landmarks.landmark[4]
            thumb_base = hand_landmarks.landmark[2]

            # Check if thumb is raised (y position lower means up)
            if thumb_tip.y < thumb_base.y:
                thumbs_up_count += 1

        # Detect two thumbs up
        if thumbs_up_count == 2:
            print("✅ Detected Two Thumbs Up! Opening Voice Assistant...")
            open_voice_assistant()

    cv2.imshow("Voice Assistant Gesture", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
