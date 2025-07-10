import cv2
import mediapipe as mp
import pyttsx3
import time
from win10toast import ToastNotifier

# Initialize Mediapipe Hands module
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Text-to-Speech Engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)  # Speed of speech
tts_engine.setProperty('volume', 1.0)  # Max volume

# Windows Notification Reader
toaster = ToastNotifier()

# Capture webcam feed
cap = cv2.VideoCapture(0)
last_notification_time = 0  # Prevent multiple triggers

def is_palm_open(landmarks):
    """
    Detects an open palm gesture.
    - All fingers should be extended (tips above knuckles).
    """
    fingers_extended = all(landmarks[i][2] < landmarks[i-2][2] for i in [8, 12, 16, 20])
    return fingers_extended

def read_notifications():
    """Reads out a dummy notification (Can be extended for real notifications)."""
    message = "You have a new message. Meeting at 3 PM. Don't forget!"
    print("🔔 Reading Notification: ", message)
    tts_engine.say(message)
    tts_engine.runAndWait()
    toaster.show_toast("Message Preview", message, duration=3)

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm_list = [(id, int(lm.x * img.shape[1]), int(lm.y * img.shape[0])) for id, lm in enumerate(hand_landmarks.landmark)]
            
            if is_palm_open(lm_list) and time.time() - last_notification_time > 3:
                print("📢 Palm Detected – Reading Notifications!")
                read_notifications()
                last_notification_time = time.time()

            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Message Preview Gesture", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
