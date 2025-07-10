import cv2
import mediapipe as mp
import time
import os
import platform

# Initialize Mediapipe Face & Hands
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
face_detection = mp_face.FaceDetection(min_detection_confidence=0.7)

# Capture webcam feed
cap = cv2.VideoCapture(0)

chin_rest_start = None  # Track time when the hand touches the chin
sleep_threshold = 3  # Time in seconds before sleep mode activates

def put_system_to_sleep():
    """Puts the system to sleep based on the OS."""
    if platform.system() == "Windows":
        os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")
    elif platform.system() == "Darwin":  # macOS
        os.system("pmset sleepnow")
    elif platform.system() == "Linux":
        os.system("systemctl suspend")

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hands_results = hands.process(img_rgb)
    face_results = face_detection.process(img_rgb)

    hand_near_chin = False
    chin_position = None

    # Detect face and chin position
    if face_results.detections:
        for detection in face_results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = img.shape
            chin_position = (int((bboxC.xmin + bboxC.width / 2) * w), int((bboxC.ymin + bboxC.height) * h + 40))

    # Detect hand position
    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            lm_list = [(id, int(lm.x * img.shape[1]), int(lm.y * img.shape[0])) for id, lm in enumerate(hand_landmarks.landmark)]

            if chin_position:
                for _, x, y in lm_list:
                    if abs(y - chin_position[1]) < 30:  # Check if hand is near chin
                        hand_near_chin = True
                        break

            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Check if hand is resting on chin for the threshold time
    if hand_near_chin:
        if chin_rest_start is None:
            chin_rest_start = time.time()
        elif time.time() - chin_rest_start >= sleep_threshold:
            print("💤 System Sleeping...")
            put_system_to_sleep()
            break  # Exit after triggering sleep mode
    else:
        chin_rest_start = None  # Reset timer if hand moves away

    cv2.imshow("Sleep Mode Gesture", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
