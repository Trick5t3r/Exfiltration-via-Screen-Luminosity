# Ce code modifie la luminosité de tout l'ordinateur
# en suivant le message binaire donné en entré
##########

import screen_brightness_control as sbc
import time

def message_to_binary(message):
    return ''.join(format(ord(char), '08b') for char in message)

def send_message_with_brightness(binary_message, max_add_brightness=10, duration=0.5):
    print(f"Message binaire : {binary_message}")

    current_brightness_initale = sbc.get_brightness()[0]
    current_brightness = current_brightness_initale
    print("Luminosité actuelle :", current_brightness)

    if current_brightness+max_add_brightness>100:
        current_brightness -=max_add_brightness

    for bit in binary_message:
        if bit == '1':
            sbc.set_brightness(current_brightness+max_add_brightness)
        else:
            sbc.set_brightness(current_brightness)
        
        time.sleep(duration)

    # Rétablit la luminosité initiale
    sbc.set_brightness(current_brightness_initale)

# Exemple de message binaire
message = "1100110011001100110011001100110011001100110011001100110011001100110011001100110011001100110011001100"

#Send message
send_message_with_brightness(message, max_add_brightness=2, duration=0.2)
