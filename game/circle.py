import cv2
import math

#####
## "Entidad" circulo que se dibuja en la pantalla
#
class Circle:

    def __init__(self, x, y, radius, color):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color
        self.active = True

        self.age = 0
        self.life_duration = 10.0
        self.spawn_time = 0

        self.state = "spawning"

        # Para efectos visuales
        self.scale = 0
        self.opacity = 1


    def draw(self, frame):
        if self.active:
            cv2.circle(
                frame,
                (self.x, self.y),
                self.radius,
                self.color,
                -1
            )

    def update(self, delta_time):

        self.age += delta_time

        if self.age >= self.life_duration:
            self.active = False


    def check_collision(self, point_x, point_y):
        distance = math.sqrt(
            (self.x - point_x) ** 2 +
            (self.y - point_y) ** 2
        )

        return distance <= self.radius