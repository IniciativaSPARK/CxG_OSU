########
## Clase encargadd de DECIDIR CUANDO SE GENERAN los circulos del juego
#
class SpawnSystem:

    def __init__(self, circle_manager):

        self.circle_manager = circle_manager

        # Lista de eventos programados
        self.spawn_events = []

        # Índice del siguiente evento
        self.current_event_index = 0

        self.load_level()

    def load_level(self):
        """
        Aquí se cargarían los eventos del nivel.
        Por ahora solo usamos una lista fija.
        """

        self.spawn_events = [

            {
                "time": 1.0,
                "x": 300,
                "y": 300
            },

            {
                "time": 5.0,
                "x": 600,
                "y": 400
            },

            {
                "time": 10,
                "x": 900,
                "y": 250
            },

            {
                "time": 20.0,
                "x": 20,
                "y": 20
            }
        ]

    def update(self, game_time):
        """
        Revisa si ya es momento de activar
        el siguiente evento programado.
        """
        
        # Si ya no quedan eventos
        if self.current_event_index >= len(self.spawn_events):
            return

        next_event = self.spawn_events[
            self.current_event_index
        ]

        # ¿Ya llegó el momento?
        if game_time >= next_event["time"]:

            self.circle_manager.spawn_circle(
                next_event["x"],
                next_event["y"]
            )

            self.current_event_index += 1