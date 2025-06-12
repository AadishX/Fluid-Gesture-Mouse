import cv2
import mediapipe as mp
import ctypes

# Set screen resolution manual
screen_width, screen_height =  2560, 1440

# Win32 mouse controls
SetCursorPos = ctypes.windll.user32.SetCursorPos
mouse_event = ctypes.windll.user32.mouse_event
LEFT_DOWN = 0x0002
LEFT_UP = 0x0004
RIGHT_DOWN = 0x0008
RIGHT_UP = 0x0010

# Smoothing factor
diddy_oil = 0.3
prev_x, prev_y = 0, 0  # Initial smoothed positions

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Webcam init
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

def is_palm_open(hand_landmarks):
    finger_tips = [8, 12, 16, 20]
    finger_mcp = [5, 9, 13, 17]
    count = sum(hand_landmarks.landmark[tip].y < hand_landmarks.landmark[mcp].y
                for tip, mcp in zip(finger_tips, finger_mcp))
    return count >= 4

def is_fist(hand_landmarks):
    finger_tips = [8, 12, 16, 20]
    finger_mcp = [5, 9, 13, 17]
    count = sum(hand_landmarks.landmark[tip].y > hand_landmarks.landmark[mcp].y
                for tip, mcp in zip(finger_tips, finger_mcp))
    return count >= 4

with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.6) as hands:
    while cap.isOpened():
        success, img = cap.read()
        if not success:
            continue

        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                label = results.multi_handedness[i].classification[0].label  # "Left" or "Right"

                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Use hand center (landmark 9)
                x_norm = hand_landmarks.landmark[9].x
                y_norm = hand_landmarks.landmark[9].y

                # Map to screen
                screen_x = int(x_norm * screen_width)
                screen_y = int(y_norm * screen_height)

                # Smooth position
                prev_x = int(diddy_oil * screen_x + (1 - diddy_oil) * prev_x)
                prev_y = int(diddy_oil * screen_y + (1 - diddy_oil) * prev_y)

                if is_palm_open(hand_landmarks):
                    SetCursorPos(prev_x, prev_y)
                elif is_fist(hand_landmarks):
                    if label == "Left":
                        mouse_event(LEFT_DOWN, 0, 0, 0, 0)
                        mouse_event(LEFT_UP, 0, 0, 0, 0)
                    elif label == "Right":
                        mouse_event(RIGHT_DOWN, 0, 0, 0, 0)
                        mouse_event(RIGHT_UP, 0, 0, 0, 0)

        cv2.imshow("Fluid Gesture Mouse", img)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
