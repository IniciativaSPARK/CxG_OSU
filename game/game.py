import cv2
import time

from game.hand_tracker import HandTracker
from game.spawn_system import SpawnSystem
from game.circle_manager import CircleManager
from game.config import *

#######
## Clase que contiene y dirige todo el juego
#
class Game:

    def __init__(self):

        self.cap: cv2.VideoCapture = None # type: ignore
        self.running = False

        self.hand_tracker = HandTracker()
        self.circle_manager = CircleManager()
        self.spawn_system = SpawnSystem(self.circle_manager)

        self.score = 0

        self.delta_time = 0
        self.last_frame_time = 0
        self.game_time = 0

        self.frame_delay = 1 / TARGET_FPS

    
    # Configuraciones iniciales del juego
    def setup(self):

        self.cap = cv2.VideoCapture(CAMERA_INDEX)

        if not self.cap.isOpened():
            raise Exception("No se pudo abrir la cámara")

        self.running = True

        self.last_frame_time = time.time()

        print("JUEGO INICIADO")

    
    # Procesa los inputs del teclado
    def process_input(self):

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            self.running = False


    # Actualiza los componentes del juego
    def update(self, delta_time, hand_positions):

        self.score = self.circle_manager.update(
            delta_time,
            hand_positions,
            self.score
        )

        self.spawn_system.update(
            self.game_time
        )

    # DIBUJA los elementos del juego en pantalla
    def render(self, frame):

        self.circle_manager.draw(frame)

        cv2.putText(
            frame,
            f"Score: {self.score}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        cv2.imshow(WINDOW_NAME, frame)

    
    # Limitar los fps segun el tiempo transcurrido
    def limitar_fps(self, _start_time):
        elapsed = time.time() - _start_time
        sleep_time = max(0, self.frame_delay - elapsed)
        time.sleep(sleep_time)


    # Inicia el juego
    def run(self):

        self.setup()

        while self.running:

            ### CALCULAR DETLA TIME ###
            current_time = time.time()

            self.delta_time = (current_time - self.last_frame_time)

            self.last_frame_time = current_time

            self.game_time += self.delta_time
            
            start_time = time.time()

            success, frame = self.cap.read()

            if not success:
                break

            ### LEER /PROCESAR INPUT ###
            frame = cv2.flip(frame, 1)

            frame, hand_positions = (
                self.hand_tracker.process_frame(frame)
            )

            self.process_input()

            ### ACTUALIZAR EL ESTADO DEL JUEGO ###
            self.update(self.delta_time, hand_positions)

            ### RENDERIZAR LA INFO EN LA PANTALLA ###
            self.render(frame)

            ### Limitar frames ###
            self.limitar_fps(start_time)

        self.destroy()


    # Termina los componentes relacionados al juego
    def destroy(self):
        self.cap.release()
        cv2.destroyAllWindows()

        print("JUEGO FINALIZADO")