import cv2
import mediapipe as mp
import pyautogui
import platform
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Capture webcam feed
cap = cv2.VideoCapture(0)

def open_accessibility_settings():
    """Opens accessibility settings based on the operating system."""
    os_name = platform.system()

    if os_name == "Windows":
        print("🛠️ Opening Windows Accessibility Settings...")
        pyautogui.hotkey("win", "u")  # Opens Windows accessibility settings
    elif os_name == "Darwin":  # macOS
        print("🛠️ Opening macOS Accessibility Settings...")
        os.system("open /System/Library/PreferencePanes/UniversalAccessPref.prefPane")
    elif os_name == "Linux":
        print("🛠️ Opening Linux Accessibility Settings...")
        os.system("gnome-control-center universal-access")  # GNOME-based systems

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            thumb_tip = hand_landmarks.landmark[4]
            thumb_base = hand_landmarks.landmark[2]

            # Detect thumbs-up (Thumb tip above base)
            if thumb_tip.y < thumb_base.y:
                print("✅ Detected Thumbs-Up Gesture! Opening Accessibility Settings...")
                open_accessibility_settings()

    cv2.imshow("Accessibility Hub Gesture", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
