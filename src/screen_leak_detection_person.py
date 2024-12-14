# Ce code copie l'écran et change la couleur des pixels selon "redSubtractorValue" (utilise opengl)
# en suivant le message binaire donné en entré
##### Parametres #####
VIDEO_PATH = None#"records/tv_4.MOV" #None # Remplacez par le chemin de votre vidéo ou mettre None pour le direct

CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4

model_cfg = "yolo_files/yolov3-tiny.cfg"
model_weights = "yolo_files/yolov3-tiny.weights"
class_names_file = "yolo_files/coco.names"

# Exemple de message
#message = "SALUT"  # Changez par le message que vous voulez
binary_message = "1100110011001100110011001100110011001100" #message_to_binary(message)
print(f"Message binaire : {binary_message}")

# Paramètres
f = 0.2  # Fréquence en secondes
redSubtractorValue = 0.1 #Valeur de modification de la couleur

########################
import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import pyautogui
from PIL import Image
from screeninfo import get_monitors
import time
import cv2



# Vertex Shader
VERTEX_SHADER = """
#version 330
in vec2 position;
in vec2 texCoords;
out vec2 fragTexCoords;

void main() {
    fragTexCoords = texCoords;
    gl_Position = vec4(position, 0.0, 1.0);
}
"""

# Fragment Shader
FRAGMENT_SHADER = """
#version 330
in vec2 fragTexCoords;
out vec4 fragColor;

uniform sampler2D screenTexture;
uniform float redSubtractor;
uniform int redBoostActive;

void main() {
    vec4 texColor = texture(screenTexture, fragTexCoords);
    if (redBoostActive == 1) {
        texColor.r = max(texColor.r - redSubtractor, 0.0); // Empêche les valeurs négatives
        texColor.b = max(texColor.b - redSubtractor, 0.0); // Empêche les valeurs négatives
        texColor.g = max(texColor.g - redSubtractor, 0.0); // Empêche les valeurs négatives
    }
    fragColor = texColor;
}
"""

def message_to_binary(message):
    return ''.join(format(ord(char), '08b') for char in message)

def detect_person(net, output_layers, class_names, frame):
    """Détecte les contours rectangulaires dans l'image fournie."""
    height, width = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    rectangles = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > CONFIDENCE_THRESHOLD and class_names[class_id] == "person":
                return True
    return False



# Fonction principale
def main():
    # Charger le modèle YOLO
    net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # Charger les noms des classes
    with open(class_names_file, "r") as file_coco:
        class_names = file_coco.read().strip().split("\n")

    if VIDEO_PATH is not None:
        cap = cv2.VideoCapture(VIDEO_PATH)
    else:
        cap = cv2.VideoCapture(0)  

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Frame rate: {fps} FPS")
    if not glfw.init():
        raise Exception("GLFW n'a pas pu être initialisé")

    glfw.window_hint(glfw.DECORATED, True)
    width, height = 3840, 2160
    window = glfw.create_window(width, height, "Real-Time Screen Capture", None, None)

    if not window:
        glfw.terminate()
        raise Exception("La fenêtre GLFW n'a pas pu être créée")

    glfw.make_context_current(window)

    vertices = np.array([
        -1, -1, 0, 0,
         1, -1, 1, 0,
         1,  1, 1, 1,
        -1,  1, 0, 1
    ], dtype=np.float32)

    indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)

    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)

    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    ebo = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

    shader = compileProgram(
        compileShader(VERTEX_SHADER, GL_VERTEX_SHADER),
        compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
    )
    glUseProgram(shader)

    position = glGetAttribLocation(shader, "position")
    glEnableVertexAttribArray(position)
    glVertexAttribPointer(position, 2, GL_FLOAT, GL_FALSE, 4 * vertices.itemsize, ctypes.c_void_p(0))

    texCoords = glGetAttribLocation(shader, "texCoords")
    glEnableVertexAttribArray(texCoords)
    glVertexAttribPointer(texCoords, 2, GL_FLOAT, GL_FALSE, 4 * vertices.itemsize, ctypes.c_void_p(2 * vertices.itemsize))

    screenTexture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, screenTexture)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    # Préparation pour OpenGL
    redBoostActiveLoc = glGetUniformLocation(shader, "redBoostActive")
    redSubtractorLoc = glGetUniformLocation(shader, "redSubtractor")


    index = 0
    last_time = time.time()
    brightness_variation_active = False
    pressed = False
    paused = False
    paused_time = time.time()
    pressed_time = time.time()
    while not glfw.window_should_close(window):
        glClear(GL_COLOR_BUFFER_BIT)

        # Capture d'écran en temps réel
        screenshot = pyautogui.screenshot()
        screenshot = screenshot.resize((width, height))  # Redimensionner pour correspondre à la fenêtre
        data = np.array(screenshot).astype(np.uint8)
        data = np.flip(data, axis=0)  # Inverser verticalement l'image pour OpenGL

        # Vérifier l'état de la touche "Entrée"
        if glfw.get_key(window, glfw.KEY_ENTER) == glfw.PRESS:
            pressed = True
            pressed_time = time.time()

            print("ENTER PRESSED")

        if pressed and time.time() > pressed_time + 2.0:
            brightness_variation_active = True
            pressed = False
            print("MESSAGE START")

        ret, frame = cap.read()
        if not ret:
            break

        #Regarde si on a detecter une personne
        if detect_person(net, output_layers, class_names, frame):
            print("PAUSED")
            paused = True
            paused_time = time.time()
        elif paused:
            print("STOP PAUSE")
            paused = False
            pressed_time -= (paused_time-time.time())
            last_time -= (paused_time-time.time())


        # Si la variation de luminosité est activée
        if brightness_variation_active:
            current_time = time.time()
            if not paused and (current_time - last_time) >= f:
                print("pass")
                last_time = current_time
                if index + 1 == len(binary_message):
                    brightness_variation_active = False
                    print("MESSAGE END")
                index = (index + 1) % len(binary_message)
                
            # Récupérer l'état actuel du bit
            current_bit = int(binary_message[index])

            # Charger les données dans la texture OpenGL
            glBindTexture(GL_TEXTURE_2D, screenTexture)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data)
            glUniform1i(redBoostActiveLoc, current_bit)
            glUniform1f(redSubtractorLoc, redSubtractorValue)

        else:
            glBindTexture(GL_TEXTURE_2D, screenTexture)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data)
        # Dessiner l'image
        glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()

if __name__ == "__main__":
    main()
