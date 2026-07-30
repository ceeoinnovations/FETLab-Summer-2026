from machine import SoftI2C, Pin, ADC
import network
import socket
import time
from ssd1306 import SSD1306_I2C

# WiFi setup
wlan = network.WLAN(network.STA_IF)
wlan.active(False)
time.sleep(0.5)
wlan.active(True)
wlan.connect("tufts_eecs", "foundedin1883")
while not wlan.isconnected():
    time.sleep(0.1)
print("WiFi connected! IP:", wlan.ifconfig()[0])

# I2C + display
i2c = SoftI2C(scl=Pin(7), sda=Pin(6))
oled = SSD1306_I2C(128, 64, i2c)
oled.fill(0)
oled.text("WiFi connected!", 0, 0)
oled.show()

# Gesture sensor setup (APDS-9960, 0x39)
i2c.writeto_mem(0x39, 0x80, bytes([0x45]))  # enable power + gesture
i2c.writeto_mem(0x39, 0xAB, bytes([0x01]))  # gesture enable
i2c.writeto_mem(0x39, 0xA3, bytes([0x3C]))  # gesture threshold

# VCNL4040 distance sensor setup (0x60)
i2c.writeto_mem(0x60, 0x03, bytes([0x00, 0x0E]))  # enable proximity

# Button Select = Pin 9
btn = Pin(9, Pin.IN, Pin.PULL_UP)

# UDP socket
MAC_IP = "10.5.11.213"
MAC_PORT = 5010
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# MIDI settings
CHANNEL = 1
INSTRUMENT = 0
sock.sendto(bytes([0xC0 | CHANNEL, INSTRUMENT]), (MAC_IP, MAC_PORT))

# Music state
NOTES = [60, 62, 64, 65, 67, 69, 71, 72]  # C D E F G A B C
NOTE_NAMES = ["C", "D", "E", "F", "G", "A", "B", "C+"]
note_index = 0
octave_shift = 0
current_note = None
velocity = 64  # default velocity

# Gesture tracking
gesture_data = []
last_active = time.ticks_ms()

def send_note_on(note, vel):
    sock.sendto(bytes([0x90 | CHANNEL, note, vel]), (MAC_IP, MAC_PORT))
    print("Note ON:", note, "vel:", vel)

def send_note_off(note):
    sock.sendto(bytes([0x80 | CHANNEL, note, 0]), (MAC_IP, MAC_PORT))
    print("Note OFF:", note)

def update_display(note_idx, octave, vel, playing):
    oled.fill(0)
    name = NOTE_NAMES[note_idx]
    oled.text("Note: " + name, 0, 0)
    oled.text("Oct:  " + str(octave), 0, 16)
    oled.text("Vel:  " + str(vel), 0, 32)
    oled.text("ON" if playing else "OFF", 0, 48)
    oled.show()

update_display(note_index, octave_shift, velocity, False)
print("Ready!")

while True:
    # --- Button + VCNL4040 → velocity ---
    if not btn.value():  # button held
        raw = i2c.readfrom_mem(0x60, 0x08, 2)
        proximity = raw[0] | (raw[1] << 8)
        velocity = int((proximity / 200) * 127)  # max real range ~200
        velocity = max(1, min(127, velocity))
        print("Button held | proximity:", proximity, "velocity:", velocity)

    # --- Gesture sensor → note/octave ---
    status = i2c.readfrom_mem(0x39, 0xAF, 1)[0]
    if status & 0x01:
        data = i2c.readfrom_mem(0x39, 0xFC, 4)
        u, d, l, r = data[0], data[1], data[2], data[3]
        if max(u, d, l, r) > 3:
            gesture_data.append((u, d, l, r))
            last_active = time.ticks_ms()
    time.sleep_ms(5)

    # analyze gesture after 300ms of inactivity
    if len(gesture_data) > 2 and time.ticks_diff(time.ticks_ms(), last_active) > 300:
        first = gesture_data[0]
        last = gesture_data[-1]
        ud_diff = (first[0] - first[1]) - (last[0] - last[1])
        lr_diff = (first[2] - first[3]) - (last[2] - last[3])

        if current_note is not None:
            send_note_off(current_note)
            current_note = None

        if abs(ud_diff) > abs(lr_diff):
            if ud_diff > 0:
                octave_shift = max(-1, octave_shift - 1)
                print("Octave down:", octave_shift)
            else:
                octave_shift = min(1, octave_shift + 1)
                print("Octave up:", octave_shift)
        else:
            if lr_diff > 0:
                note_index = (note_index + 1) % len(NOTES)
                print("Next note:", NOTE_NAMES[note_index])
            else:
                note_index = (note_index - 1) % len(NOTES)
                print("Prev note:", NOTE_NAMES[note_index])

        note = NOTES[note_index] + (octave_shift * 12)
        note = max(0, min(127, note))
        send_note_on(note, velocity)
        current_note = note
        update_display(note_index, octave_shift, velocity, True)

        # auto note off after 500ms
        time.sleep_ms(500)
        send_note_off(current_note)
        current_note = None
        update_display(note_index, octave_shift, velocity, False)

        gesture_data = []

    time.sleep(0.02)

