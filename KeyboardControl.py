import cv2
import mediapipe as mp
import ctypes
import time

# Space bar key event
KEYEVENTF_KEYDOWN = 0x0000
KEYEVENTF_KEYUP = 0x0002
VK_SPACE = 0x20

def press_space_bar():
    ctypes.windll.user32.keybd_event(VK_SPACE, 0, KEYEVENTF_KEYDOWN, 0)
    ctypes.windll.user32.keybd_event(VK_SPACE, 0, KEYEVENTF_KEYUP, 0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def is_any_pinch(hand_landmarks, threshold=0.07):
    """
    Returns True if thumb tip is close to any finger tip (index, middle, ring, pinky).
    """
    thumb_tip = hand_landmarks.landmark[4]
    finger_tips = [8, 12, 16, 20]
    for tip in finger_tips:
        finger_tip = hand_landmarks.landmark[tip]
        dist = ((thumb_tip.x - finger_tip.x) ** 2 + (thumb_tip.y - finger_tip.y) ** 2) ** 0.5
        if dist < threshold:
            return True
    return False

last_space_press = 0
SPACE_COOLDOWN = 0.7  # seconds

def draw_palm_gap_lines(img, hand_landmarks):
    pairs = [(8, 5), (12, 9), (16, 13), (20, 17)]
    h, w, _ = img.shape
    for tip, mcp in pairs:
        x1, y1 = int(hand_landmarks.landmark[tip].x * w), int(hand_landmarks.landmark[tip].y * h)
        x2, y2 = int(hand_landmarks.landmark[mcp].x * w), int(hand_landmarks.landmark[mcp].y * h)
        color = (0, 255, 0)
        cv2.line(img, (x1, y1), (x2, y2), color, 2)
        cv2.circle(img, (x1, y1), 6, (0, 255, 0), -1)
        cv2.circle(img, (x2, y2), 6, (0, 255, 0), -1)

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
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                draw_palm_gap_lines(img, hand_landmarks)

                if is_any_pinch(hand_landmarks):
                    now = time.time()
                    if now - last_space_press > SPACE_COOLDOWN:
                        press_space_bar()
                        last_space_press = now

        cv2.imshow("Fluid Gesture Mouse", img)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
