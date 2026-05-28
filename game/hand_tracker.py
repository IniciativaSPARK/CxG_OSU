import cv2
import mediapipe as mp
from collections import defaultdict
from typing import List, Tuple, Dict, Optional

class HandTracker:    
    # Constantes de clase
    PALM_POINTS = [0, 1, 5, 9, 13, 17]  # Puntos para centro de palma
    FINGER_NAMES = ["Pulgar", "indice", "Medio", "Anular", "Menique"]
    
    # Configuración de dedos: (punta, media, base)
    FINGER_CONFIG = {
        "Pulgar": (4, 3, 2),
        "indice": (8, 6, 5),
        "Medio": (12, 10, 9),
        "Anular": (16, 14, 13),
        "Menique": (20, 18, 17)
    }
    
    # Colores (BGR)
    COLOR_BLUE = (255, 0, 0)
    COLOR_GREEN = (0, 255, 0)
    COLOR_RED = (0, 0, 255)
    COLOR_WHITE = (255, 255, 255)
    
    def __init__(self, 
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 smoothing_window: int = 5,
                 smoothing_threshold: float = 0.6):
        """
        Inicializa el tracker de manos
        
        Args:
            min_detection_confidence: Confianza mínima para detección
            min_tracking_confidence: Confianza mínima para seguimiento
            smoothing_window: Ventana para suavizado de dedos
            smoothing_threshold: Umbral para considerar dedo levantado
        """
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Suavizado de dedos
        self.smoothing_window = smoothing_window
        self.smoothing_threshold = smoothing_threshold
        self._finger_history = defaultdict(list)
    
    def _get_landmark_xy(self, hand_landmarks, landmark_id: int, frame_shape: Tuple) -> Tuple[int, int]:
        """Convierte landmark normalizado a coordenadas de píxel"""
        h, w, _ = frame_shape
        lm = hand_landmarks.landmark[landmark_id]
        return (int(lm.x * w), int(lm.y * h))
    
    def get_hand_center(self, hand_landmarks, frame_shape: Tuple) -> Optional[Tuple[int, int]]:
        """Calcula el centro de la palma promediando puntos clave"""
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
        """Detecta si el pulgar está extendido (considerando lateralidad)"""
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
        """Detecta si un dedo (no pulgar) está extendido"""
        if finger_name == "Pulgar":
            return False  # El pulgar se maneja aparte
        
        tip_id, mid_id, base_id = self.FINGER_CONFIG[finger_name]
        tip = hand_landmarks.landmark[tip_id]
        mid = hand_landmarks.landmark[mid_id]
        base = hand_landmarks.landmark[base_id]
        
        # La punta debe estar más arriba que la articulación media y la base
        return tip.y < mid.y - 0.01 and tip.y < base.y - 0.01
    
    def _smooth_finger_state(self, hand_idx: int, finger_name: str, is_extended: bool) -> bool:
        """Aplica suavizado a la detección de dedos para evitar parpadeos"""
        key = f"hand{hand_idx}_{finger_name}"
        self._finger_history[key].append(1.0 if is_extended else 0.0)
        
        # Mantener solo la ventana de suavizado
        if len(self._finger_history[key]) > self.smoothing_window:
            self._finger_history[key].pop(0)
        
        avg = sum(self._finger_history[key]) / len(self._finger_history[key])
        return avg >= self.smoothing_threshold
    
    def _draw_finger_tip_circle(self, frame, hand_landmarks, finger_name: str, is_extended: bool, radius: int = 6):
        """Dibuja un círculo en la punta del dedo (color verde si extendido, rojo si no)"""
        tip_id = self.FINGER_CONFIG[finger_name][0]
        x, y = self._get_landmark_xy(hand_landmarks, tip_id, frame.shape)
        
        color = self.COLOR_GREEN if is_extended else self.COLOR_RED
        cv2.circle(frame, (x, y), radius, color, -1)
        cv2.circle(frame, (x, y), radius, self.COLOR_WHITE, 1)  # Borde blanco fino
    
    def _draw_hand_info(self, frame, hand_idx: int, hand_landmarks, is_right_hand: bool, 
                        extended_fingers: List[str]):
        """Dibuja toda la información visual de una mano"""
        # Dibujar landmarks y conexiones (estilo más limpio)
        self.mp_drawing.draw_landmarks(
            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2),
            self.mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
        )
        
        # Punto azul en el centro de la palma
        center = self.get_hand_center(hand_landmarks, frame.shape)
        if center:
            cv2.circle(frame, center, 8, self.COLOR_BLUE, -1)
            cv2.circle(frame, center, 8, self.COLOR_WHITE, 1)
        
        # Círculos en puntas de dedos
        for finger in self.FINGER_NAMES:
            is_extended = finger in extended_fingers
            self._draw_finger_tip_circle(frame, hand_landmarks, finger, is_extended, radius=8)
    
    def process_frame(self, frame):
        """
        Procesa el frame y devuelve:
        - Frame con anotaciones visuales
        - Lista de posiciones (centro de palma) para colisiones
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        hand_positions = []
        all_extended_fingers = []
        hands_info = [] 
        
        # Limpiar historiales de manos no detectadas
        active_keys = set()
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_idx, (hand_landmarks, handedness_info) in enumerate(
                zip(results.multi_hand_landmarks, results.multi_handedness)
            ):
                handedness = handedness_info.classification[0].label
                is_right = (handedness == "Right")
                
                # Detectar dedos extendidos (raw)
                extended_raw = []
                # Pulgar
                if self._is_thumb_extended(hand_landmarks, is_right):
                    extended_raw.append("Pulgar")
                # Otros dedos
                for finger in self.FINGER_NAMES[1:]:  # Excluir pulgar
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
                
                self._draw_hand_info(frame, hand_idx, hand_landmarks, is_right, extended_smoothed)
                
                # Agregar centro de palma
                center = self.get_hand_center(hand_landmarks, frame.shape)
                if center:
                    hand_positions.append(center)
        
        # Limpiar historiales de dedos que ya no están activos
        for key in list(self._finger_history.keys()):
            if key not in active_keys:
                del self._finger_history[key]
        
    
        self._draw_text_info(frame, hands_info)
        
        return frame, hand_positions
    
    def _draw_text_info(self, frame, hands_info: List[Tuple[str, List[str]]]):
        """Dibuja el texto informativo en la esquina superior"""
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