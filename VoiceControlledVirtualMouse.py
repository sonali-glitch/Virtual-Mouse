import cv2
import mediapipe as mp
import pyautogui
import speech_recognition as sr
import time

# Initialize Mediapipe Hands module
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Initialize Speech Recognizer
recognizer = sr.Recognizer()

# Capture webcam feed
cap = cv2.VideoCapture(0)

# Dragging state
dragging = False

def listen_for_command():
    """Recognize voice commands for mouse actions."""
    with sr.Microphone() as source:
        print("Listening for a command...")
        recognizer.adjust_for_ambient_noise(source)
        try:
            audio = recognizer.listen(source, timeout=3)
            command = recognizer.recognize_google(audio).lower()
            print("Heard:", command)
            return command
        except sr.UnknownValueError:
            print("Could not understand.")
        except sr.RequestError:
            print("Speech recognition service unavailable.")
        except sr.WaitTimeoutError:
            print("No command detected.")
        return None

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Listen for a command every few seconds
    command = listen_for_command()

    if command:
        if "left click" in command:
            pyautogui.click()
        elif "right click" in command:
            pyautogui.rightClick()
        elif "scroll up" in command:
            pyautogui.scroll(5)
        elif "scroll down" in command:
            pyautogui.scroll(-5)
        elif "drag" in command:
            pyautogui.mouseDown()
            dragging = True
        elif "drop" in command:
            pyautogui.mouseUp()
            dragging = False

    cv2.imshow("Voice-Controlled Virtual Mouse", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
