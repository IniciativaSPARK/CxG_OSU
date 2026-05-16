import cv2
import mediapipe as mp

# Inicializar MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Configurar la detección de manos
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Inicializar la cámara
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: No se pudo abrir la cámara")
    exit()

print("Cámara funcionando. Presiona 'q' para salir")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Error al leer el frame")
        break
    
    # Voltear imagen horizontalmente (efecto espejo)
    frame = cv2.flip(frame, 1)
    
    # Convertir BGR a RGB para MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Procesar el frame
    results = hands.process(frame_rgb)
    
    # Dibujar landmarks si hay manos detectadas
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS
            )
    
    # Mostrar el frame
    cv2.imshow('Deteccion de Manos', frame)
    
    # Salir con 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()