from machine import SoftI2C, Pin, ADC
import network
import socket
import time

# WiFi setup
wlan = network.WLAN(network.STA_IF)
wlan.active(True)
wlan.connect("tufts_eecs", "foundedin1883")

while not wlan.isconnected():
    time.sleep(0.1)
print("WiFi connected! IP:", wlan.ifconfig()[0])

# I2C setup, defined by the manufacture
i2c = SoftI2C(scl=Pin(7), sda=Pin(6))

# wake up accelerometer
i2c.writeto_mem(0x4C, 0x07, bytes([0x01]))

# potentiometer, defined on board
pot = ADC(Pin(3))

# WiFi and Raspberry IP addres
MAC_IP = "10.5.15.136"
MAC_PORT = 5010
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# base notes (middle octave)
BASE_NOTES = {
    "left":  60,  # C4
    "right": 64,  # E4
    "down":  67,  # G4
    "up":    71,  # B4
}

THRESHOLD = 10000
octave_shift = 0  # starts at middle octave
last_note = None
last_shake = False

CHANNEL = 5
INSTRUMENT = 33  # 0 = Grand Piano, change to switch instrument
sock.sendto(bytes([0xC0 | CHANNEL, INSTRUMENT]), (MAC_IP, MAC_PORT))

print("Ready!")

while True:
    try:
        # --- Potentiometer → velocity ---
        pot_val = pot.read_u16()
        velocity = int((pot_val / 65535) * 127)
        velocity = max(1, min(127, velocity))

        # --- Accelerometer → octave shift ---
        data = i2c.readfrom(0x4C, 3)
        time.sleep_ms(5)
        x_acc = data[0]
        # detect shake: value goes above 40 or below 20 (resting is around 32)
        shaking = x_acc > 40 or x_acc < 20
        if shaking and not last_shake:
            if x_acc > 40:
                octave_shift = min(1, octave_shift + 1)   # shift up
                print("Octave up! Shift:", octave_shift)
            else:
                octave_shift = max(-1, octave_shift - 1)  # shift down
                print("Octave down! Shift:", octave_shift)
        last_shake = shaking

        # --- Joystick → note ---
        joy_data = i2c.readfrom_mem(0x20, 0x03, 5)
        time.sleep_ms(5)
        x = (joy_data[0] << 8) | joy_data[1]
        y = (joy_data[2] << 8) | joy_data[3]

        if x < THRESHOLD:
            direction = "left"
        elif x > 65535 - THRESHOLD:
            direction = "right"
        elif y < THRESHOLD:
            direction = "down"
        elif y > 65535 - THRESHOLD:
            direction = "up"
        else:
            direction = None

        # calculate final note with octave shift
        if direction is not None:
            note = BASE_NOTES[direction] + (octave_shift * 12)
            note = max(0, min(127, note))  # clamp to valid MIDI range
        else:
            note = None

        # send note if changed
        if note != last_note:
            if last_note is not None:
                sock.sendto(bytes([0x80 | CHANNEL, last_note, 0]), (MAC_IP, MAC_PORT))
            if note is not None:
                sock.sendto(bytes([0x90 | CHANNEL, note, velocity]), (MAC_IP, MAC_PORT))
                print("Note:", note, "Velocity:", velocity, "Octave shift:", octave_shift)
            last_note = note

    except OSError:
        print("I2C error - retrying...")

    time.sleep(0.05)

