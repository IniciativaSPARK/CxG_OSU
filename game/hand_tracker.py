import cv2
import mediapipe as mp

############
## Clase enfocada en la deteccion de las manos y gestos
#
class HandTracker:

    def __init__(self):

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process_frame(self, frame):

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.hands.process(frame_rgb)

        hand_positions = []

        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:

                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )

                # Punta del dedo índice
                index_tip = hand_landmarks.landmark[
                    self.mp_hands.HandLandmark.INDEX_FINGER_TIP
                ]

                h, w, _ = frame.shape

                x = int(index_tip.x * w)
                y = int(index_tip.y * h)

                hand_positions.append((x, y))

                cv2.circle(frame, (x, y), 10, (255, 0, 0), -1)

        return frame, hand_positions