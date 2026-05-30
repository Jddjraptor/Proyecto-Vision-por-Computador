## Proyecto-Vision-por-Computador

## Descripción general

Este proyecto explora el uso de técnicas de **Computer Vision** para controlar videojuegos mediante señales visuales captadas por una cámara web, reemplazando parcialmente los dispositivos tradicionales de entrada. Como caso de estudio se utiliza **Super Mario Bros (NES)** ejecutado en un emulador, permitiendo controlar al personaje mediante **gestos de mano** o **movimientos oculares** detectados en tiempo real.

---

## Archivos del proyecto

### `main.py`
Implementación principal basada en **gestos de mano**. Utiliza OpenCV para detectar la apertura y cierre de la mano dentro de diferentes regiones de interés de la cámara, asociando cada región a un botón específico del juego (izquierda, derecha, salto y poder).

### `alternative_main.py`
Implementación alternativa basada en **seguimiento ocular**. Emplea MediaPipe Face Mesh para detectar la dirección de la mirada y los parpadeos, permitiendo controlar el movimiento de Mario y accionar botones mediante gestos realizados con los ojos.

### `requirements.txt`
Contiene todas las dependencias necesarias para ejecutar cualquiera de las dos implementaciones.

---

## Instalación y ejecución

### Requisitos previos

- Cámara web funcional.
- Ejecución en una **máquina local** (Windows/Linux).  
  **No se recomienda Google Colab u otros servicios en la nube**, ya que `pydirectinput` necesita interactuar directamente con el teclado del sistema operativo.
- Emulador **Mesen** instalado.
- ROM de **Super Mario Bros (NES)** cargada en el emulador.

### Pasos

1. Descargar o clonar este repositorio.
2. Descomprimir la carpeta del proyecto (si aplica).
3. Instalar las dependencias:

```bash
pip install -r requirements.txt
```

4. Abrir el emulador Mesen y cargar la ROM de Super Mario Bros.
5. Ejecutar el programa deseado:

```bash
python main.py
```

o

```bash
python alternative_main.py
```

---

## Nota sobre MediaPipe

La implementación ocular utiliza **MediaPipe**, el cual puede presentar incompatibilidades en algunas instalaciones recientes (específicamente, la versión no puede ser superior a la 0.10.21). Por este motivo, el proyecto utiliza:

```txt
mediapipe==0.10.8
```

Si la instalación falla o el programa no reconoce correctamente el módulo, se recomienda crear un entorno virtual dedicado (por ejemplo mediante **Conda**) e instalar allí las dependencias antes de ejecutar los archivos. Un flujo común consiste en crear el entorno con Conda y ejecutar posteriormente el proyecto desde un entorno de desarrollo como **Spyder** o desde la terminal asociada al entorno creado.
