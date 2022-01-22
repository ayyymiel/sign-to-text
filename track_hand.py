import cv2
import time
import mediapipe as mp


class ZaHando:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5,
                                    min_tracking_confidence=0.5)  # detect the hands
        self.mp_draw = mp.solutions.drawing_utils  # key points on hands

    def find_hando(self, get_img):
        img_rgb = cv2.cvtColor(get_img, cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(img_rgb)

        if self.result.multi_hand_landmarks:
            for hand_lms in self.result.multi_hand_landmarks:  # can ignore this, mediapipe recognizes this issue
                self.mp_draw.draw_landmarks(get_img, hand_lms, self.mp_hands.HAND_CONNECTIONS)
        return get_img

    def find_pos(self, image):
        the_list = []
        if self.result.multi_hand_landmarks:
            the_hand = self.result.multi_hand_landmarks[0]
            for id, lm, in enumerate(the_hand.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                the_list.append([id, cx, cy])
                cv2.circle(image, (cx, cy), 3, (255, 0, 255), cv2.FILLED)
        return the_list


def call():
    # initialize FPS calculators
    p_time = 0
    cap = cv2.VideoCapture(0)

    okuyasu = ZaHando()

    while True:
        success, img = cap.read()
        hand_image = okuyasu.find_hando(img)
        as_list = okuyasu.find_pos(hand_image)
        if as_list != 0:
            print(as_list[4])

        c_time = time.time()
        fps = 1 / (c_time - p_time)
        p_time = c_time
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Video Window", img)

        cv2.waitKey(1)


call()
