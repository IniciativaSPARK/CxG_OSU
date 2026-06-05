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
    def update(self, delta_time, hand_positions, is_fist: bool):

        self.score = self.circle_manager.update(
            delta_time,
            hand_positions,
            self.score,
            is_fist
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
        
        # Variables para gesto de puño
        fist_start_time = None
        fist_hold_duration = 0.5  # 0.5 segundos (hold requerido)
        fist_scored = False  # evita sumar puntos repetidos por el mismo hold


        while self.running:

            ### CALCULAR DELTA TIME ###
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

            # Ahora process_frame devuelve 3 valores
            frame, hand_positions, hand_landmarks_list = (
                self.hand_tracker.process_frame(frame)
            )

            # DETECTAR GESTO DE PUÑO SOSTENIDO
            if hand_landmarks_list:
                for hand_landmarks in hand_landmarks_list:
                    is_fist = self.hand_tracker.is_fist_gesture(hand_landmarks)
                    if is_fist:
                        if fist_start_time is None:
                            fist_start_time = time.time()
                        elif time.time() - fist_start_time >= fist_hold_duration:
                            # Eliminar SOLO el/los círculo(s) donde el centro azul está sobre el círculo.
                            # `hand_positions` contiene los centros (en el mismo orden) de las manos detectadas.
                            if not fist_scored:
                                # Eliminar SOLO círculos donde el centro azul está sobre el círculo
                                # y sumar +1 punto extra por cada eliminación con puño (hold).
                                before_count = len(self.circle_manager.circles)
                                for center in hand_positions:
                                    prev_active = [c.active for c in self.circle_manager.circles]
                                    # Llamada: la eliminación la hace CircleManager
                                    self.circle_manager.check_fist_on_circle(center, is_fist=True)
                                    after_active = [c.active for c in self.circle_manager.circles]
                                    # sumar puntos por cada círculo que pasó de active=True a active=False
                                    for i in range(min(len(prev_active), len(after_active))):
                                        if prev_active[i] and not after_active[i]:
                                            self.score += 1
                                fist_scored = True

                            fist_start_time = None
                            print("¡Puño detectado! Eliminación selectiva con centro azul.")
                    else:
                        fist_start_time = None
                        fist_scored = False

            else:
                fist_start_time = None


            self.process_input()

            ### ACTUALIZAR EL ESTADO DEL JUEGO ###
            # `is_fist` controla el temporizador de eliminación por mantener la mano sobre el círculo
            self.update(self.delta_time, hand_positions, is_fist=(fist_start_time is not None))

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