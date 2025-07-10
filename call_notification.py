import cv2
import mediapipe as mp
import pyttsx3
import time

# Initialize Mediapipe Hands module
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Initialize Text-to-Speech Engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)  # Adjust speed
tts_engine.setProperty('volume', 1.0)  # Max volume

# Call log simulation (for PC version)
last_missed_call = "John Doe at 2:30 PM"  # Mock data

# Capture webcam feed
cap = cv2.VideoCapture(0)
last_call_check = 0  # Prevent repeated triggers

def is_c_shape(landmarks):
    """
    Detects a 'C' shape gesture:
    - Thumb and index finger form a curved 'C'.
    - Middle, ring, and pinky fingers slightly bent.
    """
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    # Thumb and index should be close to forming a "C"
    thumb_index_dist = abs(thumb_tip[0] - index_tip[0]) + abs(thumb_tip[1] - index_tip[1])
    
    # Middle, Ring, and Pinky should not be fully extended
    fingers_bent = middle_tip[1] > landmarks[9][1] and ring_tip[1] > landmarks[13][1] and pinky_tip[1] > landmarks[17][1]

    return thumb_index_dist < 100 and fingers_bent

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm_list = [(id, int(lm.x * img.shape[1]), int(lm.y * img.shape[0])) for id, lm in enumerate(hand_landmarks.landmark)]
            
            if is_c_shape(lm_list) and time.time() - last_call_check > 5:
                print("📞 'C' Gesture Detected – Checking Call Log")
                call_info = f"Last missed call from {last_missed_call}"
                print("📞", call_info)
                
                # Speak the call information
                tts_engine.say(call_info)
                tts_engine.runAndWait()
                
                last_call_check = time.time()

            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Call Notification Gesture", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
