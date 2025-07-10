import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Hands module
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6)

# Capture webcam feed
cap = cv2.VideoCapture(0)

def adjust_gamma(image, gamma=1.5):
    """Apply Gamma Correction to brighten dark images."""
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def enhance_contrast(image):
    """Apply Histogram Equalization to improve contrast in low light."""
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])  # Equalize the Y channel
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

while True:
    success, img = cap.read()
    if not success:
        break

    # Preprocessing for Low-Light Conditions
    img = adjust_gamma(img, gamma=1.8)  # Adjust brightness
    img = enhance_contrast(img)  # Enhance contrast

    # Convert to RGB for Mediapipe processing
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Detection - Low Light Mode", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
