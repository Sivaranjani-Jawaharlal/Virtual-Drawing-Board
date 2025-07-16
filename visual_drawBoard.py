import cv2
import numpy as np
import mediapipe as mp

# Initialize webcam
cap = cv2.VideoCapture(0)

# Drawing canvas
canvas = None

# Initialize MediaPipe hand detector
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Function to detect finger up
def fingers_up(hand_landmarks):
    tips = [8, 12, 16, 20]
    fingers = []

    # Thumb (compare x-coordinates)
    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other fingers (compare y-coordinates)
    for tip in tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers

# Initialize previous point
prev_x, prev_y = 0, 0

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            lmList = hand_landmarks.landmark
            h, w, _ = frame.shape

            # Get index finger tip coordinates
            x = int(lmList[8].x * w)
            y = int(lmList[8].y * h)

            # Check finger status
            fingers = fingers_up(hand_landmarks)

            if fingers[1] == 1 and fingers[2] == 0:
                # Draw Mode - Only index finger up
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = x, y

                cv2.line(canvas, (prev_x, prev_y), (x, y), (0, 0, 255), 5)
                prev_x, prev_y = x, y

            else:
                prev_x, prev_y = 0, 0

            mpDraw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)

    # Combine the drawing with webcam image
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, inv_canvas = cv2.threshold(gray_canvas, 50, 255, cv2.THRESH_BINARY_INV)
    inv_canvas = cv2.cvtColor(inv_canvas, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame, inv_canvas)
    frame = cv2.bitwise_or(frame, canvas)

    # Add instructions
    cv2.putText(frame, "Draw with Index Finger. Press 'c' to Clear, 'q' to Quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("Virtual Drawing Board", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        canvas = np.zeros_like(frame)
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
