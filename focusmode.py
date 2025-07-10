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

def enable_do_not_disturb():
    """Enables Do Not Disturb mode based on OS."""
    os_name = platform.system()

    if os_name == "Windows":
        print("🔕 Enabling Focus Assist (Do Not Disturb) on Windows...")
        pyautogui.hotkey("win", "a")  # Opens the action center
        pyautogui.sleep(1)
        pyautogui.press("tab", presses=3, interval=0.5)
        pyautogui.press("enter")
    elif os_name == "Darwin":  # macOS
        print("🔕 Enabling Do Not Disturb on macOS...")
        os.system("osascript -e 'tell application \"System Events\" to key code 107 using {command down, option down}'")
    elif os_name == "Linux":
        print("🔕 Enabling Do Not Disturb on Linux...")
        os.system("gsettings set org.gnome.desktop.notifications show-banners false")

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Check if fingers are extended (Open Palm)
            fingers_extended = 0
            for i in [8, 12, 16, 20]:  # Index, Middle, Ring, and Pinky Finger Tips
                if hand_landmarks.landmark[i].y < hand_landmarks.landmark[i - 2].y:
                    fingers_extended += 1

            # Check if thumb is extended
            if hand_landmarks.landmark[4].x < hand_landmarks.landmark[2].x:  
                fingers_extended += 1

            # If all five fingers are extended, activate Do Not Disturb
            if fingers_extended == 5:
                print("✅ Detected 'Stop' Gesture! Enabling Do Not Disturb Mode...")
                enable_do_not_disturb()

    cv2.imshow("Focus Mode Gesture", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
