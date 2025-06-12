import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Start video capture
webcam = cv2.VideoCapture(0)

# Initialize the Hands model once
with mp_hands.Hands(
    max_num_hands=3,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
) as hands:

    while webcam.isOpened():
        success, img = webcam.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert image to RGB
        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the frame and find hands
        results = hands.process(img_rgb)

        # Convert image back to BGR for OpenCV
        img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # Draw hand landmarks if detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Show the image
        cv2.imshow('FluidGestureMouse', img)

        # Exit on 'q' key
        if cv2.waitKey(5) & 0xFF == ord("q"):
            break

# Release resources
webcam.release()
cv2.destroyAllWindows()