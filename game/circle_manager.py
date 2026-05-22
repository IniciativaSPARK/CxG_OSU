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
        # Funcion para eliminar a todos los circulos que fueron destruidos
        pass

    def update(self, delta_time, hand_positions, score):

        for circle in self.circles:

            circle.update(delta_time)

            if not circle.active:
                continue

            for hand_x, hand_y in hand_positions:

                if circle.check_collision(hand_x, hand_y):
                    circle.active = False
                    score += 1

        return score

    def draw(self, frame):

        for circle in self.circles:
            circle.draw(frame)