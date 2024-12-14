########################
# Ce code de test
# permet de voir si comibner des créneaux décalés
# Permet de réduire l'erreur sur la reconstruction
####### Parametres #########
# Message binaire
binary_message = "1100110011"
noise_level = 0.01 #C'est le bruit gaussien de fond
noise_shift = 1 #Int c'est le bruit sur le decalage du début de chaque créneau

############################



import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.linalg import lstsq
from optimisation_maxence import *

def pad_signals_to_max_length(signals):
    """
    Complète les signaux de manière à ce qu'ils aient tous la même longueur (la longueur du signal le plus long).
    
    Args:
        signals (list of np.ndarray): Liste des signaux à padder.
        
    Returns:
        list of np.ndarray: Liste des signaux padés.
    """
    # Trouver la longueur du signal le plus long
    max_length_signal = max(sig.shape[0] for sig in signals)
    
    padded_signals = []
    for sig in signals:
        if sig.shape[0] < max_length_signal:
            sig_padded = np.pad(sig, (0, max_length_signal - sig.shape[0]), 'constant')
        else:
            sig_padded = sig
        padded_signals.append(sig_padded)
    
    return padded_signals

def reconstruct_square_wave(signals, fps = 50, freq_message = 0.2):
    """
    Reconstruit un signal carré à partir de signaux bruités et décalés.
    
    Args:
        signals (list of np.ndarray): Liste de signaux bruités.
        sampling_rate (float): Taux d'échantillonnage en Hz.
        
    Returns:
        np.ndarray: Signal carré reconstruit.
    """
    sampling_rate = int(fps * freq_message)
    signals = pad_signals_to_max_length(signals)
    ref_signal = signals[0]  # Signal de référence
    aligned_signals = []
    lags= []
    for signal in signals:
        correlation = correlate(signal, ref_signal, mode='full')
        lag = np.argmax(correlation) - len(ref_signal) + 1
        lags.append(lag)
        aligned_signal = np.roll(signal, -lag)
        aligned_signals.append(aligned_signal)

    aligned_signals = np.array(aligned_signals)
    averaged_signal = np.mean(aligned_signals, axis=0)

    #threshold = (np.max(averaged_signal) + np.min(averaged_signal)) / 2
    square_wave = least_squares_reconstruction(aligned_signals) #(averaged_signal > threshold).astype(float)

    return signals, square_wave, averaged_signal

def least_squares_reconstruction(aligned_signals):
    """
    Utilise les moindres carrés pour reconstruire un signal carré à partir de signaux bruités et alignés.
    
    Args:
        aligned_signals (list of np.ndarray): Liste des signaux alignés.
        
    Returns:
        np.ndarray: Signal reconstruit utilisant les moindres carrés.
    """
    averaged_signal = np.mean(aligned_signals, axis=0)
    mess, cren = decode_message(averaged_signal)

    print(mess)
    return cren


def generate_square_wave_from_binary(message, decalage_initiale, freq_message =0.2, fps = 50, noise_level=0.01, noise_shift=1):
    """
    Génère un signal carré bruité à partir d'un message binaire.
    
    Args:
        message (str): Message binaire (ex. "10101").
        sampling_rate (int): Nombre de points par seconde.
        noise_level (float): Amplitude du bruit ajouté.
        
    Returns:
        np.ndarray: Signal carré généré.
    """
    sampling_rate = int(freq_message * fps)
    taille_tab = len(message)
    t = np.linspace(0, taille_tab, taille_tab * sampling_rate * 5)  # Temps total
    square_wave = np.zeros_like(t)
    segment_length = sampling_rate 
    total_length = decalage_initiale

    for i, bit in enumerate(message):
        start = total_length  + np.random.randint(-noise_shift, noise_shift)
        start = max(0, start)
        end = start + segment_length + np.random.randint(-noise_shift, noise_shift)
        end = min(end, t.shape[0])
        total_length += end-start
        if bit == '1':
            square_wave[start:end] = 1

    # Ajout de bruit
    noisy_signal = square_wave + np.random.normal(0, noise_level, size=square_wave.shape)
    return noisy_signal
if __name__ == "__main__":

    # Génération de signaux bruités décalés
    noisy_signals = [
        generate_square_wave_from_binary(binary_message, shift, noise_level=noise_level, noise_shift=noise_shift) for shift in range(100, 190, 30)  # Décalages multiples de 50 points
    ]


    # Reconstruction
    noisy_signals, reconstructed_wave, averaged_signal = reconstruct_square_wave(noisy_signals)

    
    max_length_signal = noisy_signals[0].shape[0]

    # Visualisation
    t = np.arange(max_length_signal)  # Temps total
    plt.figure(figsize=(12, 6))
    plt.subplot(3, 1, 1)
    for signal in noisy_signals:
        plt.plot(t, signal, alpha=0.6)
    plt.title("Signaux bruités et décalés")
    
    plt.subplot(3, 1, 2)
    plt.plot(t, averaged_signal, label="Signal moyenné")
    plt.plot(t, reconstructed_wave, label="Signal carré reconstruit", color='red')
    plt.title("Signal carré reconstruit")
    plt.tight_layout()
    plt.show()