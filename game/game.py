import cv2
import time
import webbrowser

from game.hand_tracker import HandTracker
from game.spawn_system import SpawnSystem
from game.circle_manager import CircleManager
from game.config import *

MENU = "menu"
PLAYING = "playing"

YOUTUBE_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

BUTTON_X1, BUTTON_Y1 = 490, 310
BUTTON_X2, BUTTON_Y2 = 790, 410

class Game:

    def __init__(self):
        self.cap = None
        self.running = False
        self.state = MENU

        self.hand_tracker = HandTracker()
        self.circle_manager = CircleManager()
        self.spawn_system = SpawnSystem(self.circle_manager)

        self.score = 0
        self.delta_time = 0
        self.last_frame_time = 0
        self.game_time = 0
        self.frame_delay = 1 / TARGET_FPS

    def setup(self):
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self.cap.isOpened():
            raise Exception("No se pudo abrir la cámara")
        self.running = True
        self.last_frame_time = time.time()
        print("JUEGO INICIADO")

    def process_input(self):
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.running = False

   def ambos_dedos_en_boton(self, hand_positions):
    if len(hand_positions) < 1:
        return False
    for (x, y) in hand_positions:
        if BUTTON_X1 < x < BUTTON_X2 and BUTTON_Y1 < y < BUTTON_Y2:
            return True
    return False

    def update(self, delta_time, hand_positions):
        if self.state == MENU:
            if self.ambos_dedos_en_boton(hand_positions):
                self.state = PLAYING
                webbrowser.open(YOUTUBE_URL)

        elif self.state == PLAYING:
            self.score = self.circle_manager.update(
                delta_time, hand_positions, self.score
            )
            self.spawn_system.update(self.game_time)

    def render(self, frame):
        if self.state == MENU:
            # Fondo semitransparente en el botón
            overlay = frame.copy()
            cv2.rectangle(overlay, (BUTTON_X1, BUTTON_Y1), (BUTTON_X2, BUTTON_Y2), (0, 200, 0), -1)
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
            # Borde del botón
            cv2.rectangle(frame, (BUTTON_X1, BUTTON_Y1), (BUTTON_X2, BUTTON_Y2), (0, 255, 0), 3)
            # Texto del botón
            cv2.putText(frame, "INICIAR", (545, 375), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            # Instrucción
            cv2.putText(frame, "Pon ambos dedos indice sobre el boton", (200, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        elif self.state == PLAYING:
            self.circle_manager.draw(frame)
            cv2.putText(frame, f"Score: {self.score}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow(WINDOW_NAME, frame)

    def limitar_fps(self, _start_time):
        elapsed = time.time() - _start_time
        sleep_time = max(0, self.frame_delay - elapsed)
        time.sleep(sleep_time)

    def run(self):
        self.setup()
        while self.running:
            current_time = time.time()
            self.delta_time = current_time - self.last_frame_time
            self.last_frame_time = current_time
            self.game_time += self.delta_time
            start_time = time.time()

            success, frame = self.cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            frame, hand_positions = self.hand_tracker.process_frame(frame)
            self.process_input()
            self.update(self.delta_time, hand_positions)
            self.render(frame)
            self.limitar_fps(start_time)

        self.destroy()

    def destroy(self):
        self.cap.release()
        cv2.destroyAllWindows()
        print("JUEGO FINALIZADO")