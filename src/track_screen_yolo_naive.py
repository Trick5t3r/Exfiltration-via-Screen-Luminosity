# Ce code track les écran, définie une zone de pertience et creer un objet "Rectangle"
# pour tracer chacun des écrans en parallèle
# puis calcul une valuer pour chaque écran (la luminosite ou une cetaine couleur moyenne) et la stock
# puis en choisissant lequel ecran on veut traiter, en fait une analyse et recupere le message binaire
##### Parametres #####
video_path = None #"records/VID_5.mp4" # Remplacez par le chemin de votre vidéo ou mettre None pour le direct
scale = 0.5  # Ajustez ce facteur pour la taille de l'image
percentil_selct = 10

padding_factor = 0.3 # doit etre <0.5

# Chargement des fichiers nécessaires
model_cfg = "yolo_files/yolov3-tiny.cfg"
model_weights = "yolo_files/yolov3-tiny.weights"
class_names_file = "yolo_files/coco.names"
######################

import cv2
import numpy as np
import matplotlib.pyplot as plt

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

    
def decode_signal(X,Y,threeshold_up,threeshold_down,window_size,freq_signal):
    message = ""
    Y_smoothed = moving_average(Y[10:], window_size=window_size)
    X = X[10:]
    Y_diff = np.diff(Y_smoothed)
    Y_diff /=np.max(np.abs(Y_diff))
    X_diff = X[window_size:]

    plt.plot(X_diff, Y_diff)
    plt.title(f'diff')
    plt.show()
    
    is_zero = False
    is_one = False

    for idx,dlumi in enumerate(Y_diff):

        if dlumi > threeshold_up and is_zero == False and is_one == False:
            
            is_zero = False
            is_one = True
            time_start = X_diff[idx]
        
        if dlumi < threeshold_down and is_one:
            delta_time = X_diff[idx] - time_start
            n = int(delta_time/freq_signal) - 1
            message += n*"1"
            is_zero = True
            is_one = False
            time_start = X_diff[idx]

        if dlumi > threeshold_up and is_zero:
            delta_time = X_diff[idx] - time_start
            n = int(delta_time/freq_signal) - 1
            message += n*"0"
            is_one = True
            is_zero = False
            time_start = X_diff[idx]


    return message[1:]

next_id = 0

class RectangleObject:
    def __init__(self, contour):
        global next_id
        self.id = next_id 
        next_id += 1

        self.contour = contour
        self.center = self.calculate_center()
        self.area = cv2.contourArea(contour)
        self.is_tracked = True  # Pour indiquer si le rectangle est suivi

        self.brightness = []
        self.frame_activated = []

    def calculate_center(self):
        moments = cv2.moments(self.contour)
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            return (cx, cy)
        return None

    def update(self, new_contour):
        """Mise à jour de la position et des propriétés du rectangle."""
        self.contour = new_contour
        self.center = self.calculate_center()
        self.area = cv2.contourArea(new_contour)
        self.is_tracked = True

    def distance_to(self, other_center):
        """Calcul de la distance entre deux centres de rectangle."""
        if self.center is None or other_center is None:
            return float('inf')
        return np.linalg.norm(np.array(self.center) - np.array(other_center))


def detect_rectangles(frame):
    """Détecte les contours rectangulaires dans l'image fournie."""
    height, width = frame.shape[:2]

    # Prétraitement pour YOLO
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    rectangles = []

    # Analyse des sorties YOLO
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filtrer les objets avec une confiance suffisante et la classe 'tvmonitor'
            if confidence > CONFIDENCE_THRESHOLD and class_names[class_id] == "tvmonitor":
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype("int")
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                # Crée une boîte englobante
                box = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
                new_rect = RectangleObject(box)
                add_rect = True
                for rect in rectangles:
                    if rect.distance_to(new_rect.center) < 50:
                        add_rect = False
                        break
                if add_rect:
                    rectangles.append(new_rect)
    return rectangles


def track_rectangles(prev_rectangles, curr_rectangles, max_distance=50):
    """Associe les rectangles entre les frames pour un suivi efficace."""
    for prev_rect in prev_rectangles:
        prev_rect.is_tracked = False  # Marque tous les rectangles comme non suivis initialement

    for curr_rect in curr_rectangles:
        closest_prev_rect = None
        min_distance = max_distance

        # Trouve le rectangle le plus proche de la frame précédente
        for prev_rect in prev_rectangles:
            distance = prev_rect.distance_to(curr_rect.center)
            if distance < min_distance:
                min_distance = distance
                closest_prev_rect = prev_rect

        # Si un rectangle correspondant est trouvé, on met à jour sa position
        if closest_prev_rect:
            closest_prev_rect.update(curr_rect.contour)
            curr_rect.is_tracked = False

    # Retourne les rectangles mis à jour et non mis à jour (perdus)
    return prev_rectangles + [rect for rect in curr_rectangles if rect.is_tracked is True]

def generate_padded_contour(contour, padding_factor=0.2):
    """
    Génère un contour rectangulaire à partir des coordonnées avec padding.

    Args:
        x_start: Coordonnée x du coin supérieur gauche.
        y_start: Coordonnée y du coin supérieur gauche.
        x_end: Coordonnée x du coin inférieur droit.
        y_end: Coordonnée y du coin inférieur droit.

    Returns:
        Un contour sous forme de tableau numpy.
    """
    # Extraire la bounding box du contour
    x, y, w, h = cv2.boundingRect(contour)

    # Ajouter un padding de 10% à chaque côté
    pad_x = int(w * padding_factor)
    pad_y = int(h * padding_factor)

    # Calculer les nouvelles coordonnées avec padding
    x_start = max(x + pad_x, 0)
    y_start = max(y + pad_y, 0)
    x_end = min(x + w - pad_x, frame.shape[1])
    y_end = min(y + h - pad_y, frame.shape[0])

    # Générer les sommets du rectangle
    contour = np.array([
        [x_start, y_start],  # Haut-gauche
        [x_end, y_start],    # Haut-droit
        [x_end, y_end],      # Bas-droit
        [x_start, y_end]     # Bas-gauche
    ])
    return contour


def calculate_average_brightness(frame, contour, padding_factor=0.2):
    """
    Calcule la luminosité moyenne à l'intérieur d'un contour avec un padding.

    Args:
        frame: L'image d'entrée (frame de la vidéo).
        contour: Le contour pour lequel calculer la luminosité.
        padding_factor: Le facteur de padding (par défaut 10%).

    Returns:
        La luminosité moyenne à l'intérieur du contour avec padding.
    """

    # Crée un masque vide pour la bounding box avec padding
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    # Décale le contour pour qu'il corresponde à la région de la ROI
    shifted_contour = generate_padded_contour(contour, padding_factor) 
    
    # Remplit le masque avec le contour décalé
    cv2.drawContours(mask, [shifted_contour], -1, 255, thickness=cv2.FILLED)

    # Convertit la ROI en niveaux de gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Sélectionne les pixels à l'intérieur du contour (là où le masque est blanc)
    selected_pixels = gray[mask == 255]

    # Calcule et retourne la luminosité moyenne
    if len(selected_pixels) > 0:
        average_brightness = selected_pixels[selected_pixels>=np.percentile(selected_pixels, percentil_selct)].mean()
    else:
        average_brightness = 0  # Valeur par défaut si aucun pixel n'est sélectionné

    return average_brightness



# Charger les noms des classes
with open(class_names_file, "r") as f:
    class_names = f.read().strip().split("\n")

# Charger le modèle YOLO
net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Définir les paramètres
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4

if video_path is not None:
    cap = cv2.VideoCapture(video_path)
else:
    cap = cv2.VideoCapture(0)  

# Obtenir les couches de sortie YOLO
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Frame rate: {fps} FPS")

prev_rectangles = []

indice_frame = -1
while True:
    indice_frame += 1
    ret, frame = cap.read()
    if not ret:
        break

    
    frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))

     # Détection des rectangles dans la frame actuelle
    curr_rectangles = detect_rectangles(frame)

    # Suivi des rectangles entre les frames
    prev_rectangles = track_rectangles(prev_rectangles, curr_rectangles)

    # Affiche les rectangles suivis
    for rect in prev_rectangles:
        if rect.is_tracked and rect.center:
            rect.brightness.append(calculate_average_brightness(frame, rect.contour, padding_factor))
            rect.frame_activated.append(indice_frame)
            #cv2.circle(frame, rect.center, 5, (255, 0, 0), -1)
            cv2.drawContours(frame, [generate_padded_contour(rect.contour, padding_factor)], -1, (0, 255, 255), thickness=cv2.FILLED)
            cv2.drawContours(frame, [rect.contour], -1, (0, 255, 0), 2)
            
            cv2.putText(frame, f'ID: {rect.id} \n B:{rect.brightness[-1]}', rect.center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow("Rectangle Tracking", frame)

    # Quitte la boucle si la touche 'q' est appuyée
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if len(prev_rectangles) != 0:
    if len(prev_rectangles) == 1:
        index = 0
    else:
        index = int(input(f"Entrez un index entre 0 et {prev_rectangles[-1].id}: "))

    for rect_x in prev_rectangles:
        if rect_x.id == index:
            rect = rect_x
    if len(rect.brightness) >0:
        plt.plot(np.array(rect.frame_activated)/fps, rect.brightness)
        plt.title(f'ID {rect.id}')
        plt.show()

        decoded_message = decode_signal(np.array(rect.frame_activated)/fps,rect.brightness,0.05,-0.05,4,0.2)
        print("Message décodé : ", decoded_message)
