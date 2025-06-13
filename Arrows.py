import cv2
import mediapipe as mp
import ctypes
import time

# Key codes for arrow keys
VK_LEFT = 0x25
VK_RIGHT = 0x27
VK_UP = 0x26
VK_DOWN = 0x28
KEYEVENTF_KEYDOWN = 0x0000
KEYEVENTF_KEYUP = 0x0002

def press_key(vk_code):
    ctypes.windll.user32.keybd_event(vk_code, 0, KEYEVENTF_KEYDOWN, 0)
    ctypes.windll.user32.keybd_event(vk_code, 0, KEYEVENTF_KEYUP, 0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

PALM_DEBOUNCE = 0.7     # seconds
PINCH_THRESHOLD = 0.08  # fraction of frame width
last_left_time = 0
last_right_time = 0
last_left_pinch_time = 0
last_right_pinch_time = 0

def is_palm_closed(hand_landmarks):
    # All finger tips below their MCPs (fist)
    finger_tips = [8, 12, 16, 20]
    finger_mcp = [5, 9, 13, 17]
    count = sum(hand_landmarks.landmark[tip].y > hand_landmarks.landmark[mcp].y
                for tip, mcp in zip(finger_tips, finger_mcp))
    return count >= 4

def is_pinch(hand_landmarks, img_shape):
    # Returns True if thumb tip and index tip are close
    h, w, _ = img_shape
    x1 = hand_landmarks.landmark[4].x * w
    y1 = hand_landmarks.landmark[4].y * h
    x2 = hand_landmarks.landmark[8].x * w
    y2 = hand_landmarks.landmark[8].y * h
    distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    return distance < PINCH_THRESHOLD * w, (int(x1), int(y1), int(x2), int(y2))

def draw_pretty_landmarks(img, hand_landmarks):
    h, w, _ = img.shape
    line_color = (57, 255, 20)  # Neon green
    connections = mp_hands.HAND_CONNECTIONS
    for connection in connections:
        start_idx, end_idx = connection
        x1, y1 = int(hand_landmarks.landmark[start_idx].x * w), int(hand_landmarks.landmark[start_idx].y * h)
        x2, y2 = int(hand_landmarks.landmark[end_idx].x * w), int(hand_landmarks.landmark[end_idx].y * h)
        cv2.line(img, (x1, y1), (x2, y2), line_color, 4, cv2.LINE_AA)
    for idx, lm in enumerate(hand_landmarks.landmark):
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (x, y), 7, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1, cv2.LINE_AA)

with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.6) as hands:
    while cap.isOpened():
        success, img = cap.read()
        if not success:
            continue

        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        h, w, _ = img.shape

        hand_landmarks_list = []
        hand_labels = []

        if results.multi_hand_landmarks and results.multi_handedness:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                draw_pretty_landmarks(img, hand_landmarks)
                hand_landmarks_list.append(hand_landmarks)
                label = results.multi_handedness[i].classification[0].label  # "Left" or "Right"
                hand_labels.append(label)

            now = time.time()
            for hand_landmarks, label in zip(hand_landmarks_list, hand_labels):
                # Palm close for left/right arrow
                if is_palm_closed(hand_landmarks):
                    if label == "Left" and (now - last_left_time) > PALM_DEBOUNCE:
                        press_key(VK_LEFT)
                        last_left_time = now
                        cv2.putText(img, "Left Palm Close! (Left Arrow)", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
                    elif label == "Right" and (now - last_right_time) > PALM_DEBOUNCE:
                        press_key(VK_RIGHT)
                        last_right_time = now
                        cv2.putText(img, "Right Palm Close! (Right Arrow)", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
                # Pinch for up/down arrow
                pinch, (x1, y1, x2, y2) = is_pinch(hand_landmarks, img.shape)
                if label == "Left" and pinch and (now - last_left_pinch_time) > PALM_DEBOUNCE:
                    press_key(VK_UP)
                    last_left_pinch_time = now
                    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 6, cv2.LINE_AA)
                    cv2.putText(img, "Left Pinch! (Up Arrow)", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
                elif label == "Right" and pinch and (now - last_right_pinch_time) > PALM_DEBOUNCE:
                    press_key(VK_DOWN)
                    last_right_pinch_time = now
                    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 6, cv2.LINE_AA)
                    cv2.putText(img, "Right Pinch! (Down Arrow)", (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)

        cv2.imshow("Arrow Palm/Pinch Control", img)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()