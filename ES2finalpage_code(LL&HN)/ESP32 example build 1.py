from machine import ADC, Pin
import network
import socket
import time
import gc

# WiFi setup
wlan = network.WLAN(network.STA_IF)
wlan.active(False)
time.sleep(0.5)
wlan.active(True)
wlan.connect("tufts_eecs", "foundedin1883")
while not wlan.isconnected():
    time.sleep(0.1)
gc.collect()
print("WiFi connected!")

# Light sensor
light = ADC(Pin(32))
light.atten(ADC.ATTN_11DB)

# Force sensor
fsr = ADC(Pin(33))
fsr.atten(ADC.ATTN_11DB)

# Buttons
btn1 = Pin(13, Pin.IN, Pin.PULL_UP)
btn2 = Pin(14, Pin.IN, Pin.PULL_UP)
btn3 = Pin(27, Pin.IN, Pin.PULL_UP)
btn4 = Pin(26, Pin.IN, Pin.PULL_UP)

# Socket
MAC_IP = "10.5.11.213"
MAC_PORT = 5010
addr = (MAC_IP, MAC_PORT)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
gc.collect()

channel = 6
INSTRUMENT = 8  # 0 = Grand Piano
notes_dark =   [45, 48, 50, 52]  # A3 C4 D4 E4  (buttons 1-4 in dark, one octave lower)
notes_normal = [64, 65, 67, 64]  # E4 F4 G4 E4  (buttons 1-4 in light)
pressed = [False, False, False, False]
buttons = [btn1, btn2, btn3, btn4]

sock.sendto(bytes([0xC0 | channel, INSTRUMENT]), addr)
print("Ready! Instrument set to:", INSTRUMENT)

last_mode = None

while True:
    light_val = light.read()
    if light_val < 500:
        notes = notes_dark
        mode = "DARK"
    else:
        notes = notes_normal
        mode = "NORMAL"

    if mode != last_mode:
        print("Mode:", mode)
        last_mode = mode

    fsr_val = fsr.read_u16()
    if fsr_val > 20000:  # not pressed — resting value is ~22000
        velocity = 20
    else:
        velocity = int(((20000 - fsr_val) / 20000) * 127)
        velocity = max(20, min(127, velocity))

    for i in range(4):
        if not buttons[i].value() and not pressed[i]:
            sock.sendto(bytes([0x90 | channel, notes[i], velocity]), addr)
            print("Button", i + 1, "pressed  -> Note ON", notes[i], "vel", velocity, "(", mode, ")")
            pressed[i] = True
        elif buttons[i].value() and pressed[i]:
            sock.sendto(bytes([0x80 | channel, notes[i], 0]), addr)
            print("Button", i + 1, "released -> Note OFF", notes[i])
            pressed[i] = False

    time.sleep(0.02)
