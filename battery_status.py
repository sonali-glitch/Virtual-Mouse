import cv2
import mediapipe as mp
import pyttsx3
import psutil
import time

# Initialize Mediapipe Hands module
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Initialize Text-to-Speech Engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)  # Adjust speed
tts_engine.setProperty('volume', 1.0)  # Max volume

# Capture webcam feed
cap = cv2.VideoCapture(0)
last_battery_check = 0  # Prevent repeated triggers

def get_battery_status():
    """Fetches battery percentage and charging status."""
    battery = psutil.sensors_battery()
    percentage = battery.percent
    charging = battery.power_plugged

    status = f"Battery is at {percentage} percent."
    if charging:
        status += " The laptop is charging."
    else:
        status += " The laptop is not charging."
    
    return status

def is_thumbs_up(landmarks):
    """
    Detects thumbs-up gesture:
    - Thumb extended (tip above lower joints)
    - Other fingers folded (tip below lower joints)
    """
    thumb_extended = landmarks[4][2] < landmarks[3][2]  # Thumb tip above first joint
    other_fingers_folded = all(landmarks[i][2] > landmarks[i-2][2] for i in [8, 12, 16, 20])

    return thumb_extended and other_fingers_folded

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm_list = [(id, int(lm.x * img.shape[1]), int(lm.y * img.shape[0])) for id, lm in enumerate(hand_landmarks.landmark)]
            
            if is_thumbs_up(lm_list) and time.time() - last_battery_check > 3:
                print("🔋 Thumbs Up Detected – Checking Battery Status")
                battery_status = get_battery_status()
                print("🔋", battery_status)
                
                # Speak battery status
                tts_engine.say(battery_status)
                tts_engine.runAndWait()
                
                last_battery_check = time.time()

            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Battery Status Gesture", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
