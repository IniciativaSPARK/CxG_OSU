import cv2
import mediapipe as mp
from collections import defaultdict
from typing import List, Tuple, Optional

class HandTracker:    
    # Constantes de clase
    PALM_POINTS = [0, 1, 5, 9, 13, 17]
    FINGER_NAMES = ["Pulgar", "Indice", "Medio", "Anular", "Menique"]
    
    FINGER_CONFIG = {
        "Pulgar": (4, 3, 2),
        "Indice": (8, 6, 5),
        "Medio": (12, 10, 9),
        "Anular": (16, 14, 13),
        "Menique": (20, 18, 17)
    }
    
    COLOR_BLUE = (255, 0, 0)
    COLOR_GREEN = (0, 255, 0)
    COLOR_RED = (0, 0, 255)
    COLOR_WHITE = (255, 255, 255)
    
    def __init__(self, 
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 smoothing_window: int = 5,
                 smoothing_threshold: float = 0.6):
        
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        self.smoothing_window = smoothing_window
        self.smoothing_threshold = smoothing_threshold
        self._finger_history = defaultdict(list)
    
    def _get_landmark_xy(self, hand_landmarks, landmark_id: int, frame_shape: Tuple) -> Tuple[int, int]:
        h, w, _ = frame_shape
        lm = hand_landmarks.landmark[landmark_id]
        return (int(lm.x * w), int(lm.y * h))
    
    def get_hand_center(self, hand_landmarks, frame_shape: Tuple) -> Optional[Tuple[int, int]]:
        if not hand_landmarks:
            return None
        
        h, w, _ = frame_shape
        points = []
        
        for point_id in self.PALM_POINTS:
            lm = hand_landmarks.landmark[point_id]
            points.append((int(lm.x * w), int(lm.y * h)))
        
        center_x = sum(p[0] for p in points) // len(points)
        center_y = sum(p[1] for p in points) // len(points)
        
        return (center_x, center_y)
    
    def _is_thumb_extended(self, hand_landmarks, is_right_hand: bool) -> bool:
        tip = hand_landmarks.landmark[4]
        mid = hand_landmarks.landmark[3]
        base = hand_landmarks.landmark[2]
        
        height_ok = tip.y < mid.y - 0.015 and tip.y < base.y - 0.01
        
        if is_right_hand:
            horizontal_ok = tip.x > base.x
        else:
            horizontal_ok = tip.x < base.x
            
        return height_ok and horizontal_ok
    
    def _is_finger_extended(self, hand_landmarks, finger_name: str) -> bool:
        if finger_name == "Pulgar":
            return False
        
        if finger_name not in self.FINGER_CONFIG:
            return False
            
        tip_id, mid_id, base_id = self.FINGER_CONFIG[finger_name]
        tip = hand_landmarks.landmark[tip_id]
        mid = hand_landmarks.landmark[mid_id]
        base = hand_landmarks.landmark[base_id]
        
        return tip.y < mid.y - 0.01 and tip.y < base.y - 0.01
    
    def _smooth_finger_state(self, hand_idx: int, finger_name: str, is_extended: bool) -> bool:
        key = f"hand{hand_idx}_{finger_name}"
        self._finger_history[key].append(1.0 if is_extended else 0.0)
        
        if len(self._finger_history[key]) > self.smoothing_window:
            self._finger_history[key].pop(0)
        
        avg = sum(self._finger_history[key]) / len(self._finger_history[key])
        return avg >= self.smoothing_threshold
    
    def _draw_finger_tip_circle(self, frame, hand_landmarks, finger_name: str, is_extended: bool, radius: int = 6):
        tip_id = self.FINGER_CONFIG[finger_name][0]
        x, y = self._get_landmark_xy(hand_landmarks, tip_id, frame.shape)
        
        color = self.COLOR_GREEN if is_extended else self.COLOR_RED
        cv2.circle(frame, (x, y), radius, color, -1)
        cv2.circle(frame, (x, y), radius, self.COLOR_WHITE, 1)
    
    def _draw_hand_info(self, frame, hand_landmarks, is_right_hand: bool, extended_fingers: List[str]):
        self.mp_drawing.draw_landmarks(
            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2),
            self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
        )
        
        center = self.get_hand_center(hand_landmarks, frame.shape)
        if center:
            cv2.circle(frame, center, 8, self.COLOR_BLUE, -1)
            cv2.circle(frame, center, 8, self.COLOR_WHITE, 1)
        
        for finger in self.FINGER_NAMES:
            is_extended = finger in extended_fingers
            self._draw_finger_tip_circle(frame, hand_landmarks, finger, is_extended, radius=8)
    
    def is_fist_gesture(self, hand_landmarks) -> bool:
        """Detecta gesto de puño (todos los dedos cerrados)"""
        tips = [4, 8, 12, 16, 20]
        bases = [2, 6, 10, 14, 18]
        
        extended_count = 0
        
        for tip_id, base_id in zip(tips, bases):
            tip_y = hand_landmarks.landmark[tip_id].y
            base_y = hand_landmarks.landmark[base_id].y
            
            if tip_y < base_y - 0.05:
                extended_count += 1
        
        return extended_count <= 1
    
    def process_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        hand_positions = []
        hand_landmarks_list = []
        all_extended_fingers = []
        hands_info = []
        active_keys = set()
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_idx, (hand_landmarks, handedness_info) in enumerate(
                zip(results.multi_hand_landmarks, results.multi_handedness)
            ):
                hand_landmarks_list.append(hand_landmarks)
                
                handedness = handedness_info.classification[0].label
                is_right = (handedness == "Right")
                
                extended_raw = []
                
                if self._is_thumb_extended(hand_landmarks, is_right):
                    extended_raw.append("Pulgar")
                
                for finger in self.FINGER_NAMES[1:]:
                    if self._is_finger_extended(hand_landmarks, finger):
                        extended_raw.append(finger)
                
                extended_smoothed = []
                for finger in self.FINGER_NAMES:
                    is_extended = finger in extended_raw
                    if self._smooth_finger_state(hand_idx, finger, is_extended):
                        extended_smoothed.append(finger)
                    active_keys.add(f"hand{hand_idx}_{finger}")
                
                all_extended_fingers.append(extended_smoothed)
                hands_info.append((handedness, extended_smoothed))
                
                self._draw_hand_info(frame, hand_landmarks, is_right, extended_smoothed)
                
                center = self.get_hand_center(hand_landmarks, frame.shape)
                if center:
                    hand_positions.append(center)
        
        for key in list(self._finger_history.keys()):
            if key not in active_keys:
                del self._finger_history[key]
        
        self._draw_text_info(frame, hands_info)
        
        return frame, hand_positions, hand_landmarks_list
    
    def _draw_text_info(self, frame, hands_info: List[Tuple[str, List[str]]]):
        y_offset = 40

        if not hands_info:
            cv2.putText(frame, "No se detectan manos", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.COLOR_RED, 1, cv2.LINE_AA)
            return
        
        for handedness, fingers in hands_info:
            hand_text = f"{handedness}: "
            if fingers:
                hand_text += ", ".join(fingers)
            else:
                hand_text += "Ninguno"
            
            cv2.putText(frame, hand_text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.COLOR_GREEN, 1, cv2.LINE_AA)
            y_offset += 25