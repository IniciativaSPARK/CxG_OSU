import time
from game.circle import Circle
from game.config import (
    CIRCLE_RADIUS,
    CIRCLE_COLOR
)

######
## Clase encargada de crear y eliminar circulos en la logica del juego
#
class CircleManager:

    def __init__(self):
        self.circles: list[Circle] = []
        self.interaction_timers = {}  # Guardar tiempo de interacción por círculo

    def spawn_circle(self, x, y):
        circle = Circle(
            x,
            y,
            CIRCLE_RADIUS,
            CIRCLE_COLOR
        )
        self.circles.append(circle)

    def check_collisions(self):
        pass

    def cleanup(self):
        """Elimina círculos inactivos"""
        self.circles = [c for c in self.circles if c.active]

    def clear_all_circles(self):
        """Elimina todos los círculos (para gesto de puño)"""
        self.circles.clear()
        self.interaction_timers.clear()
        print("Todos los círculos eliminados")

    def check_fist_on_circle(self, hand_center, is_fist: bool) -> None:
        """Elimina SOLO el(los) círculo(s) cuyo contorno contenga el centro de la mano,
        siempre que `is_fist` sea True.

        Args:
            hand_center: Tupla (x, y) del centro de la mano (punto azul) o None.
            is_fist: True si se detecta el gesto de puño.
        """
        if not is_fist:
            return
        if not hand_center:
            return

        hand_x, hand_y = hand_center
        # Marcar como inactivos los círculos tocados por el centro de la mano.
        for circle in self.circles:
            if circle.active and circle.check_collision(hand_x, hand_y):
                circle.active = False
                if circle.id in self.interaction_timers:
                    del self.interaction_timers[circle.id]


    def update(self, delta_time, hand_positions, score, is_fist: bool):
        """Actualiza círculos.

        - Solo mientras `is_fist` sea True: si la mano (centro azul) se mantiene sobre el círculo
          durante X segundos (2.0), el círculo se elimina.
        - Si `is_fist` es False, no se acumula el timer (se resetea).
        """
        current_time = time.time()
        
        for circle in self.circles:
            # Actualizar tiempo de vida del círculo
            circle.update(delta_time)
            
            if not circle.active:
                continue

            # Si no hay puño, reseteamos el timer de interacción de este círculo
            if not is_fist:
                if circle.id in self.interaction_timers:
                    del self.interaction_timers[circle.id]
                continue

            # Verificar colisiones con manos
            collision_detected = False
            for hand_x, hand_y in hand_positions:
                if circle.check_collision(hand_x, hand_y):
                    collision_detected = True

                    # Si es primera vez que se toca, registrar tiempo
                    if circle.id not in self.interaction_timers:
                        self.interaction_timers[circle.id] = current_time

                    # Calcular tiempo transcurrido
                    elapsed = current_time - self.interaction_timers[circle.id]

                    # Si pasaron 2 segundos, eliminar el círculo
                    if elapsed >= 2.0:
                        circle.active = False
                        score += 1
                        if circle.id in self.interaction_timers:
                            del self.interaction_timers[circle.id]
                        print(f"¡Círculo eliminado después de 2 segundos! Score: {score}")
                    
                    break  # Un círculo solo puede ser tocado por una mano
            
            # Si no hay colisión, resetear timer de este círculo
            if not collision_detected and circle.id in self.interaction_timers:
                del self.interaction_timers[circle.id]
        
        # Limpiar círculos inactivos
        self.cleanup()
        
        return score

    def draw(self, frame):
        for circle in self.circles:
            circle.draw(frame)