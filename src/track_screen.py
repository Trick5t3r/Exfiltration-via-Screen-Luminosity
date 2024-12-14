########################
# Ce code permet de suivre les rectangles d'une video
# et de calculer le luminosite moyenne
# Puis d'afficher une representation graphique temporelle
# Ainsi que les sauts (derivees secondes)
####### Parametres ######
video_path = "records/cam_2.MOV"  #Mettre None pour la camera

#########################

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
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    rectangles = []
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4 and cv2.contourArea(contour) > 1000:
            rectangles.append(RectangleObject(approx))

    return rectangles


def track_rectangles(prev_rectangles, curr_rectangles, max_distance=50):
    """Associe les rectangles entre les frames pour un suivi efficace."""
    for prev_rect in prev_rectangles:
        prev_rect.is_tracked = False 

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

    return prev_rectangles + [rect for rect in curr_rectangles if rect.is_tracked is True]

def calculate_average_brightness(frame, contour):
    """ Calcule la luminosite moyenne a l'interieur du contour sur la frame"""
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    selected_pixels = gray[mask == 255]
    
    
    if len(selected_pixels) > 0:
        average_brightness = selected_pixels.mean()
    else:
        average_brightness = 0

    return average_brightness


if video_path is not None:
    cap = cv2.VideoCapture(video_path)
else:
    cap = cv2.VideoCapture(1)  # Remplacez par un chemin vidéo si nécessaire

fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Frame rate: {fps} FPS")

prev_rectangles = []

indice_frame = -1
while True:
    indice_frame += 1
    ret, frame = cap.read()
    if not ret:
        break

    scale = 0.25 
    frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))

    curr_rectangles = detect_rectangles(frame)

    # Suivi des rectangles entre les frames
    prev_rectangles = track_rectangles(prev_rectangles, curr_rectangles)

    # Affiche les rectangles suivis
    for rect in prev_rectangles:
        if rect.is_tracked and rect.center:
            cv2.drawContours(frame, [rect.contour], -1, (0, 255, 0), 2)
            rect.brightness.append(calculate_average_brightness(frame, rect.contour))
            rect.frame_activated.append(indice_frame)
            #cv2.circle(frame, rect.center, 5, (255, 0, 0), -1)
            cv2.putText(frame, f'ID: {rect.id} \n B:{rect.brightness[-1]}', rect.center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow("Rectangle Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

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

