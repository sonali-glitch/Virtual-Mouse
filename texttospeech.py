import cv2
import mediapipe as mp
import pyautogui
import pytesseract
import pyttsx3
import numpy as np

# Configure Tesseract OCR path (Change if needed)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Initialize Text-to-Speech engine
tts_engine = pyttsx3.init()

# Capture webcam feed
cap = cv2.VideoCapture(0)

def capture_screen():
    """Captures a screenshot of the screen."""
    screenshot = pyautogui.screenshot()
    screenshot = np.array(screenshot)
    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
    return screenshot

def extract_text_from_image(img):
    """Extracts text using OCR."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return text.strip()

def speak_text(text):
    """Converts text to speech."""
    if text:
        print(f"🗣️ Reading: {text}")
        tts_engine.say(text)
        tts_engine.runAndWait()
    else:
        print("⚠️ No text detected.")

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            index_tip = hand_landmarks.landmark[8]
            index_base = hand_landmarks.landmark[6]

            # If index finger is raised
            if index_tip.y < index_base.y:
                print("✅ Detected Raised Index Finger! Capturing Text...")
                screen_img = capture_screen()
                extracted_text = extract_text_from_image(screen_img)
                speak_text(extracted_text)

    cv2.imshow("Text-to-Speech Gesture", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
