import cv2
import time
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)  # detect the hands
mp_draw = mp.solutions.drawing_utils  # key points
cap = cv2.VideoCapture(0)
p_time = 0
c_time = 0

while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)
    if result.multi_hand_landmarks:
        for hand_lms in result.multi_hand_landmarks:
            for id, lm, in enumerate(hand_lms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)
            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)

    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Video Window", img)

    cv2.waitKey(1)
