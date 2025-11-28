import numpy as np
import cv2
import math
import time
import pydirectinput

def mantener_tecla(tecla, tiempo):
    pydirectinput.keyDown(tecla)
    time.sleep(tiempo)   # tiempo en segundos
    pydirectinput.keyUp(tecla)

def procesar_rectangulo(frame, x1, y1, x2, y2, tecla, tiempo):
    """
    Procesa un rectángulo de detección de mano.
    Si se detectan >4 defectos convexos, presiona la tecla indicada.
    """
    
    # Obtener los datos de la mano desde la subventana definida por el rectángulo
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 0)
    crop_image = frame[y1:y2, x1:x2]

    # Aplicar desenfoque Gaussiano
    blur = cv2.GaussianBlur(crop_image, (3, 3), 0)

    # Cambiar el espacio de color de BGR -> HSV
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # Crear una imagen binaria donde el blanco representará piel y el resto será negro
    mask2 = cv2.inRange(hsv, np.array([2, 0, 0]), np.array([20, 255, 255]))

    # Kernel para transformaciones morfológicas
    kernel = np.ones((5, 5))

    # Aplicar transformaciones morfológicas para eliminar ruido del fondo
    dilation = cv2.dilate(mask2, kernel, iterations=1)
    erosion = cv2.erode(dilation, kernel, iterations=1)

    # Aplicar desenfoque Gaussiano y umbralización
    filtered = cv2.GaussianBlur(erosion, (3, 3), 0)
    ret, thresh = cv2.threshold(filtered, 127, 255, 0)

    # Encontrar contornos
    # image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, hierachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    try:
        # Encontrar el contorno con el área máxima
        contour = max(contours, key=lambda x: cv2.contourArea(x))

        # Crear un rectángulo delimitador alrededor del contorno
        bx, by, bw, bh = cv2.boundingRect(contour)
        cv2.rectangle(crop_image, (bx, by), (bx + bw, by + bh), (0, 0, 255), 0)

        # Encontrar el casco convexo (convex hull)
        hull = cv2.convexHull(contour)

        # Dibujar el contorno
        drawing = np.zeros(crop_image.shape, np.uint8)
        cv2.drawContours(drawing, [contour], -1, (0, 255, 0), 0)
        cv2.drawContours(drawing, [hull], -1, (0, 0, 255), 0)

        # Calcular los defectos de convexidad
        hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull)

        # Usar la regla del coseno para calcular el ángulo en el punto más lejano
        # entre los puntos inicial y final (punta de los dedos)
        count_defects = 0

        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])

            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            angle = (math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180) / 3.14

            # Si el ángulo es menor o igual a 90 dibujar un círculo en el punto más lejano
            if angle <= 90:
                count_defects += 1
                cv2.circle(crop_image, far, 1, [0, 0, 255], -1)

            cv2.line(crop_image, start, end, [0, 255, 0], 2)

        # Presionar ESPACIO si se cumple la condición
        if count_defects >= 4:
            mantener_tecla(tecla, tiempo)
            cv2.putText(frame, f"{tecla.upper()}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 2)

    except:
        pass

#-----------------------------

# Abrir la cámara
capture = cv2.VideoCapture(0)

while capture.isOpened():
    
    # Capturar frames de la cámara
    ret, frame = capture.read()

    # Voltear horizontalmente (corrección de espejo)
    frame = cv2.flip(frame, 1)
    
    # --- RECTÁNGULO 1 (izquierda) ---
    #procesar_rectangulo(frame, 50, 10, 250, 210, "left", 0.3)

    # --- RECTÁNGULO 2 (derecha) ---
    #procesar_rectangulo(frame, 50, 250, 250, 450, "right", 0.3)
    
    # --- RECTÁNGULO 3 (salto) ---
    procesar_rectangulo(frame, 400, 10, 600, 210, "x", 0.01)
    
    # --- RECTÁNGULO 4 (poder) ---
    #procesar_rectangulo(frame, 400, 250, 600, 450, "z", 0.01)
    
    # Mostrar las imágenes necesarias
    cv2.imshow("Gesto", frame)

    # Cerrar la cámara si se presiona 'q'
    if cv2.waitKey(1) == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
