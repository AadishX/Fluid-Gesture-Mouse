import cv2
import mediapipe as mp
import ctypes
import time
import webbrowser

# --- Open Netflix in browser ---
def open_netflix_browser():
    url = "https://www.netflix.com/browse"
    webbrowser.open(url)

# Key codes for media control (Windows virtual key codes)
VK_SPACE = 0x20
VK_LEFT = 0x25
VK_RIGHT = 0x27
KEYEVENTF_KEYDOWN = 0x0000
KEYEVENTF_KEYUP = 0x0002

def press_key(vk_code):
    ctypes.windll.user32.keybd_event(vk_code, 0, KEYEVENTF_KEYDOWN, 0)
    ctypes.windll.user32.keybd_event(vk_code, 0, KEYEVENTF_KEYUP, 0)

mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

PINCH_THRESHOLD = 0.08  # for gestures
CLAP_THRESHOLD = 0.13   # fraction of frame width
DEBOUNCE = 0.7

last_play_time = 0
last_ff_time = 0
last_clap_time = 0

def is_pinch(hand_landmarks, img_shape):
    h, w, _ = img_shape
    x1 = hand_landmarks.landmark[4].x * w
    y1 = hand_landmarks.landmark[4].y * h
    x2 = hand_landmarks.landmark[8].x * w
    y2 = hand_landmarks.landmark[8].y * h
    distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    return distance < PINCH_THRESHOLD * w, (int(x1), int(y1), int(x2), int(y2))

def draw_pretty_landmarks(img, hand_landmarks):
    h, w, _ = img.shape
    line_color = (57, 255, 20)
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

# Open Netflix in browser
open_netflix_browser()
time.sleep(2)  # Give browser time to open

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

            # --- Right hand pinch: Play/Pause (Space) ---
            for hand_landmarks, label in zip(hand_landmarks_list, hand_labels):
                pinch, (x1, y1, x2, y2) = is_pinch(hand_landmarks, img.shape)
                if label == "Right" and pinch and (now - last_play_time) > DEBOUNCE:
                    press_key(VK_SPACE)
                    last_play_time = now
                    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 255), 6, cv2.LINE_AA)
                    cv2.putText(img, "Right Pinch: Play/Pause", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)

            # --- Left hand pinch: Fast Forward (Right Arrow) ---
            for hand_landmarks, label in zip(hand_landmarks_list, hand_labels):
                pinch, (x1, y1, x2, y2) = is_pinch(hand_landmarks, img.shape)
                if label == "Left" and pinch and (now - last_ff_time) > DEBOUNCE:
                    press_key(VK_RIGHT)
                    last_ff_time = now
                    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 6, cv2.LINE_AA)
                    cv2.putText(img, "Left Pinch: Fast Forward", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)

            # --- Clap (both palms close together): Rewind (Left Arrow) ---
            if len(hand_landmarks_list) == 2 and (now - last_clap_time) > DEBOUNCE:
                # Use palm center (landmark 9) for both hands
                x1 = int(hand_landmarks_list[0].landmark[9].x * w)
                y1 = int(hand_landmarks_list[0].landmark[9].y * h)
                x2 = int(hand_landmarks_list[1].landmark[9].x * w)
                y2 = int(hand_landmarks_list[1].landmark[9].y * h)
                dist = abs(x1 - x2)
                if dist < CLAP_THRESHOLD * w:
                    press_key(VK_LEFT)
                    last_clap_time = now
                    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 8, cv2.LINE_AA)
                    cv2.putText(img, "Clap: Rewind", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)

        cv2.imshow("Netflix Gesture Control", img)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()