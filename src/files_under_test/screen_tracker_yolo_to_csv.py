# Ce code permet de prendre une video, d'extraire le signal de l'écran 
# et de l'enregistrer dans un fichier csv
##### Parametres #####
name = "all_005"

video_path = f"./V2/sample/{name}.MOV" #None # Remplacez par le chemin de votre vidéo ou mettre None pour le direc
scale = 0.25  # Ajustez ce facteur pour la taille de l'image

padding_factor = 0.2 # doit etre <0.5
percentile = 40


# Chargement des fichiers nécessaires
model_cfg = "yolo_files/yolov3-tiny.cfg"
model_weights = "yolo_files/yolov3-tiny.weights"
class_names_file = "yolo_files/coco.names"
######################

import cv2
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import time
import csv

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

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
    x, y, w, h = cv2.boundingRect(contour)

    pad_x = int(w * padding_factor)
    pad_y = int(h * padding_factor)

    x_start = max(x + pad_x, 0)
    y_start = max(y + pad_y, 0)
    x_end = min(x + w - pad_x, frame.shape[1])
    y_end = min(y + h - pad_y, frame.shape[0])

    contour = np.array([
        [x_start, y_start],  
        [x_end, y_start],    
        [x_end, y_end],      
        [x_start, y_end]     
    ])
    return contour


def calculate_average_brightness(frame, contour, padding_factor=0.2, percentile = 50):
    """
    Calcule la luminosité moyenne à l'intérieur d'un contour avec un padding.

    Args:
        frame: L'image d'entrée (frame de la vidéo).
        contour: Le contour pour lequel calculer la luminosité.
        padding_factor: Le facteur de padding (par défaut 10%).

    Returns:
        La luminosité moyenne à l'intérieur du contour avec padding.
    """
    
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    shifted_contour = generate_padded_contour(contour, padding_factor) 
    
    cv2.drawContours(mask, [shifted_contour], -1, 255, thickness=cv2.FILLED)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    selected_pixels_percentile = gray[(mask == 255)]
    
    percentile_value =  np.percentile(selected_pixels_percentile, percentile)
    selected_pixels = gray[(mask == 255) & (gray >= percentile_value)]
    selected_pixels_rgb = frame[(mask == 255) & (gray >= percentile_value)]
    if len(selected_pixels) > 0:
        average_brightness = selected_pixels.mean()
        avg_r = np.mean(selected_pixels_rgb[:, 2])  # Canal Rouge (R) est à l'indice 2
        avg_g = np.mean(selected_pixels_rgb[:, 1])  # Canal Vert (G) est à l'indice 1
        avg_b = np.mean(selected_pixels_rgb[:, 0])  # Canal Bleu (B) est à l'indice 0
    else:
        average_brightness = 0  # Valeur par défaut si aucun pixel n'est sélectionné

    return [average_brightness, avg_r, avg_g, avg_b]


# Charger les noms des classes
with open(class_names_file, "r") as f:
    class_names = f.read().strip().split("\n")

# Charger le modèle YOLO
net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

# Définir les paramètres
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4

if video_path is not None:
    cap = cv2.VideoCapture(video_path)
else:
    cap = cv2.VideoCapture(0)  

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Frame rate: {fps} FPS")
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * scale)
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale)

# Initialisation du VideoWriter
output_filename = f"./records/tracking_output/tracking_{name}.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
prev_rectangles = []

indice_frame = -1
t_start = time.time()
while True:
    indice_frame += 1
    ret, frame = cap.read()
    if not ret:
        break

    
    frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))

    curr_rectangles = detect_rectangles(frame)

    prev_rectangles = track_rectangles(prev_rectangles, curr_rectangles)

    for rect in prev_rectangles:
        if rect.is_tracked and rect.center:
            rect.brightness.append(calculate_average_brightness(frame, rect.contour, padding_factor, percentile))
            rect.frame_activated.append(indice_frame)
            #cv2.circle(frame, rect.center, 5, (255, 0, 0), -1)
            cv2.drawContours(frame, [generate_padded_contour(rect.contour, padding_factor)], -1, (0, 255, 255), thickness=cv2.FILLED)
            cv2.drawContours(frame, [rect.contour], -1, (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {rect.id} \n B:{rect.brightness[-1][0]}', rect.center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow("Rectangle Tracking", frame)
    out.write(frame)
    # Quitte la boucle si la touche 'q' est appuyée
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(time.time() - t_start)

if len(prev_rectangles) != 0:
    if len(prev_rectangles) == 1:
        index = 0
    else:
        index = int(input(f"Entrez un index entre 0 et {prev_rectangles[-1].id}: "))

    for rect_x in prev_rectangles:
        if rect_x.id == index:
            rect = rect_x

    if len(rect.brightness) > 0:
        array_brightness = np.array(rect.brightness)
        time = np.array(rect.frame_activated) / fps 

        # Créer des traces Plotly pour chaque couleur
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=time, y=array_brightness[:, 0]-np.mean(array_brightness[:, 0]), mode='lines', name='Luminosité moyenne', line=dict(color='black')))
        fig.add_trace(go.Scatter(x=time, y=array_brightness[:, 1]-np.mean(array_brightness[:, 1]), mode='lines', name='Luminosité rouge', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=time, y=array_brightness[:, 2]-np.mean(array_brightness[:, 2]), mode='lines', name='Luminosité verte', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=time, y=array_brightness[:, 3]-np.mean(array_brightness[:, 3]), mode='lines', name='Luminosité bleue', line=dict(color='blue')))

        # Ajouter le titre et les labels
        fig.update_layout(
            title=f"Luminosité pour l'ID {rect.id}",
            xaxis_title="Temps (s)",
            yaxis_title="Luminosité",
            legend_title="Canaux"
        )


        # Afficher le graphique interactif
        fig.show()
inout_sec = float(input(f"Entrez le temps de départ du signal à traiter : "))

nstart = int(inout_sec*fps)

def save_data_to_csv(filename, x, array_brightness):
    """
    Enregistre les données dans un fichier CSV.
    :param filename: Le nom du fichier où enregistrer les données.
    :param x: Le vecteur des temps (array).
    :param array_brightness: Le tableau contenant les intensités lumineuses (array).
    """
    # Calculer la moyenne et soustraire de chaque colonne de array_brightness
    y = np.column_stack([array_brightness[:, i] - np.mean(array_brightness[:, i]) for i in range(array_brightness.shape[1])])

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Time', 'Brightness_1', 'Brightness_2', 'Brightness_3', 'Brightness_4'])
        

        for i in range(len(x)):
            row = [x[i]] + list(y[i])
            writer.writerow(row)
    print(f'Données enregistrées dans le fichier {filename}')

filename = f"./V2/luminosity_signal/{name}.csv"
save_data_to_csv(filename, time[nstart:], array_brightness[nstart:])