import cv2
import mediapipe as mp
import time
import os

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Capture webcam feed
cap = cv2.VideoCapture(0)

blink_start = None  # Track blinking time
blink_threshold = 0.3  # Blink duration in seconds
blue_light_filter_enabled = False

def toggle_blue_light_filter():
    """Enables or disables the blue light filter (Windows Only)."""
    global blue_light_filter_enabled
    if blue_light_filter_enabled:
        os.system("powershell.exe (Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightnessMethods).WmiSetBrightness(1, 100)")
        print("🟢 Blue Light Filter Disabled")
    else:
        os.system("powershell.exe (Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightnessMethods).WmiSetBrightness(1, 50)")
        print("🔵 Blue Light Filter Enabled")
    
    blue_light_filter_enabled = not blue_light_filter_enabled

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    left_eye_closed = False
    right_eye_closed = False

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get key eye landmark points
            left_eye_top = face_landmarks.landmark[159].y
            left_eye_bottom = face_landmarks.landmark[145].y
            right_eye_top = face_landmarks.landmark[386].y
            right_eye_bottom = face_landmarks.landmark[374].y

            # Calculate eye closure ratio
            left_eye_ratio = abs(left_eye_top - left_eye_bottom)
            right_eye_ratio = abs(right_eye_top - right_eye_bottom)

            # Check if both eyes are closed
            if left_eye_ratio < 0.02 and right_eye_ratio < 0.02:
                left_eye_closed = True
                right_eye_closed = True

    # Detect quick blink
    if left_eye_closed and right_eye_closed:
        if blink_start is None:
            blink_start = time.time()
        elif time.time() - blink_start < blink_threshold:
            print("🔵 Activating Blue Light Filter...")
            toggle_blue_light_filter()
            blink_start = None
    else:
        blink_start = None

    cv2.imshow("Eye Comfort Mode Gesture", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
