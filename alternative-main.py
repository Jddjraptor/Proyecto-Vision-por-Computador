import cv2    
import numpy as np        
import mediapipe as mp    
import pydirectinput       
import time                

# CONFIGURACIÓN TECLAS
pydirectinput.PAUSE = 0

# MEDIAPIPE FACE MESH
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,            # Solo se detecta una cara
    refine_landmarks=True,      # Activa landmarks del iris
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# LANDMARKS DEL IRIS

# Puntos que conforman el iris izquierdo
LEFT_IRIS = [474, 475, 476, 477]
# Puntos que conforman el iris derecho
RIGHT_IRIS = [469, 470, 471, 472]
# Extremos horizontales del ojo izquierdo
LEFT_EYE_LEFT = 33
LEFT_EYE_RIGHT = 133
# Extremos horizontales del ojo derecho
RIGHT_EYE_LEFT = 362
RIGHT_EYE_RIGHT = 263

# CONTROL DE TECLAS PRESIONADAS

# Evitan llamar keyDown continuamente
left_pressed = False
right_pressed = False
# Dirección previa para suavizado
prev_direction = "CENTER"

# APERTURA DE CÁMARA
cap = cv2.VideoCapture(0)

# CALCULAR CENTRO DEL IRIS
def get_iris_center(landmarks, iris_ids, w, h):
    # Promedio de coordenadas x
    x = np.mean([landmarks[i].x for i in iris_ids]) * w
    # Promedio de coordenadas y
    y = np.mean([landmarks[i].y for i in iris_ids]) * h
    return x, y

# Variable para suavizado de mirada
prev_ratio = 0
# Factor de suavizado exponencial
alpha = 0.7

# CONFIGURACIÓN DE PARPADEO

# Umbral para considerar ojo cerrado
blink_threshold = 0.05
# Tiempo mínimo entre activaciones
blink_cooldown = 0.35
# Registro del último parpadeo
last_blink_time_left = 0
last_blink_time_right = 0

# LANDMARKS PARA EAR
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# CÁLCULO DEL EYE ASPECT RATIO (EAR)
def eye_aspect_ratio(landmarks, eye_points, w, h):
    # Punto superior del ojo
    p1 = np.array([landmarks[eye_points[1]].x * w,
        landmarks[eye_points[1]].y * h])
    # Punto inferior del ojo
    p2 = np.array([landmarks[eye_points[5]].x * w,
        landmarks[eye_points[5]].y * h])
    # Extremo izquierdo
    p3 = np.array([landmarks[eye_points[0]].x * w,
        landmarks[eye_points[0]].y * h])
    # Extremo derecho
    p4 = np.array([landmarks[eye_points[3]].x * w,
        landmarks[eye_points[3]].y * h])
    # Distancia vertical
    vertical = np.linalg.norm(p1 - p2)
    # Distancia horizontal
    horizontal = np.linalg.norm(p3 - p4)
    # Relación de apertura del ojo
    return vertical / horizontal

# BUCLE PRINCIPAL
while True:
    # Leer frame de la cámara
    ret, frame = cap.read()
    if not ret:
        break

    # Efecto espejo
    frame = cv2.flip(frame, 1)
    # Dimensiones del frame
    h, w, _ = frame.shape
    # Convertir a RGB para MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Detectar rostro
    results = face_mesh.process(rgb)
    # Estado por defecto
    direction = "CENTER"

    if results.multi_face_landmarks:
        # Obtener landmarks faciales
        landmarks = results.multi_face_landmarks[0].landmark

        # PARPADEO OJO IZQUIERDO → SALTO
        left_ear = eye_aspect_ratio(landmarks,LEFT_EYE,w,h)
        if left_ear < blink_threshold:
            if time.time() - last_blink_time_left > blink_cooldown:
                pydirectinput.keyDown("s")
                time.sleep(0.15)
                pydirectinput.keyUp("s")
                last_blink_time_left = time.time()

        # PARPADEO OJO DERECHO → PODER
        right_ear = eye_aspect_ratio(landmarks,RIGHT_EYE,w,h)
        if right_ear < blink_threshold:
            if time.time() - last_blink_time_right > blink_cooldown:
                pydirectinput.keyDown("z")
                time.sleep(0.15)
                pydirectinput.keyUp("z")
                last_blink_time_right = time.time()

        # IRIS IZQUIERDO
        lx, ly = get_iris_center(landmarks,LEFT_IRIS,w,h)
        eye_lx = landmarks[LEFT_EYE_LEFT].x * w
        eye_rx = landmarks[LEFT_EYE_RIGHT].x * w
        # Posición relativa dentro del ojo
        left_ratio = (lx - eye_lx) / (eye_rx - eye_lx)

        # IRIS DERECHO
        rx, ry = get_iris_center(landmarks,RIGHT_IRIS,w,h)
        eye_lx_r = landmarks[RIGHT_EYE_LEFT].x * w
        eye_rx_r = landmarks[RIGHT_EYE_RIGHT].x * w
        right_ratio = (rx - eye_lx_r) / (eye_rx_r - eye_lx_r)

        # Decisión de mirada
        avg_ratio = (left_ratio + right_ratio) / 2
        # Suavizado exponencial
        avg_ratio = alpha * prev_ratio + (1 - alpha) * avg_ratio
        prev_ratio = avg_ratio
        # Clasificación de dirección
        if avg_ratio < 0.42:
            direction = "LEFT"
        elif avg_ratio > 0.58:
            direction = "RIGHT"
        else:
            direction = "CENTER"

        # Visualización
        cv2.putText(frame,direction,(50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        # Mostrar centros de iris
        cv2.circle(frame, (int(lx), int(ly)), 3, (0,0,255), -1)
        cv2.circle(frame, (int(rx), int(ry)), 3, (0,0,255), -1)

    # CONTROL DE MOVIMIENTO
    if direction == "LEFT":
        if not left_pressed:
            pydirectinput.keyDown("left")
            left_pressed = True
        if right_pressed:
            pydirectinput.keyUp("right")
            right_pressed = False
    elif direction == "RIGHT":
        if not right_pressed:
            pydirectinput.keyDown("right")
            right_pressed = True
        if left_pressed:
            pydirectinput.keyUp("left")
            left_pressed = False
    else:
        # Soltar teclas si mira al centro
        if left_pressed:
            pydirectinput.keyUp("left")
            left_pressed = False
        if right_pressed:
            pydirectinput.keyUp("right")
            right_pressed = False

    # Mostrar Video
    cv2.imshow("Eye Control - Mario", frame)
    # Salir con Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# LIBERACIÓN DE RECURSOS

# Asegurar que ninguna tecla quede presionada
pydirectinput.keyUp("left")
pydirectinput.keyUp("right")
# Cerrar cámara
cap.release()
# Cerrar ventanas OpenCV
cv2.destroyAllWindows()
