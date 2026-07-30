import time
import machine
import network
import socket

# -------------------------------------------------------
# WIFI SETUP
# -------------------------------------------------------
wlan = network.WLAN(network.STA_IF)
wlan.active(True)
wlan.connect("tufts_eecs", "foundedin1883")

print("Connecting to WiFi...")
while not wlan.isconnected():
    time.sleep(0.1)
print("WiFi connected:", wlan.ifconfig())

# -------------------------------------------------------
# SIMPLE UDP MIDI (sends raw MIDI messages to the Mac bridge)
# Run midi_bridge.py on the Mac; it forwards these to virtual MIDI ports.
# -------------------------------------------------------
MAC_IP = "10.42.0.1"   # <-- your Mac's IP (run: ipconfig getifaddr en0)
MIDI_PORT = 5010         # must match UDP_PORT in midi_bridge.py

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
midi_addr = socket.getaddrinfo(MAC_IP, MIDI_PORT)[0][-1]

def note_on(note, velocity=127):
    sock.sendto(bytes([0x90, note, velocity]), midi_addr)

def note_off(note):
    sock.sendto(bytes([0x80, note, 0]), midi_addr)

# -------------------------------------------------------
# PITCH BEND
# 14-bit value 0..16383, center 8192 = no bend. GarageBand applies it
# automatically to sounding notes (default range +/- 2 semitones).
# -------------------------------------------------------
BEND_CENTER = 8192       # no bend
BEND_STEP   = 256        # how far each encoder detent moves the bend

bend_value = BEND_CENTER
bend_dirty = False       # set by the encoder IRQ, sent from the main loop

def pitch_bend(value):
    value = max(0, min(16383, value))    # clamp to legal range
    sock.sendto(bytes([0xE0, value & 0x7F, (value >> 7) & 0x7F]), midi_addr)

# -------------------------------------------------------
# DRUMS
# General MIDI puts percussion on channel 10 (index 9), so the status byte
# is 0x90 | 9 = 0x99. On that channel each NOTE NUMBER is a different drum:
#   36 = kick, 38 = snare, 42 = closed hi-hat, 46 = open hi-hat, 49 = crash.
# The bridge routes channel 10 to the separate "Pico Drums" port.
# -------------------------------------------------------
DRUM_CH = 9              # MIDI channel 10, zero-indexed
KICK = 36

def drum_hit(note=KICK, velocity=120):
    sock.sendto(bytes([0x90 | DRUM_CH, note, velocity]), midi_addr)

# -------------------------------------------------------
# MAX7219 LED MATRIX
# -------------------------------------------------------
class MAX7219:
    def __init__(self, spi, cs):
        self.spi = spi
        self.cs = cs
        self.cs.value(1)
        self.buffer = [[0]*8 for _ in range(8)]
        self._write(0x09, 0x00)
        self._write(0x0A, 0x02)
        self._write(0x0B, 0x07)
        self._write(0x0C, 0x01)
        self._write(0x0F, 0x00)

    def _write(self, reg, data):
        self.cs.value(0)
        self.spi.write(bytes([reg, data]))
        self.cs.value(1)

    def fill(self, val):
        for r in range(8):
            for c in range(8):
                self.buffer[r][c] = val

    def pixel(self, x, y, val):
        if 0 <= x < 8 and 0 <= y < 8:
            self.buffer[y][x] = val

    def show(self):
        for row in range(8):
            byte = 0
            for col in range(8):
                if self.buffer[row][col]:
                    byte |= (1 << (7 - col))
            self._write(row + 1, byte)

    def brightness(self, val):
        self._write(0x0A, val & 0x0F)

spi = machine.SoftSPI(
    baudrate=1000000,
    polarity=0,
    phase=0,
    sck=machine.Pin(6),
    mosi=machine.Pin(7),
    miso=machine.Pin(4)
)
cs = machine.Pin(16, machine.Pin.OUT)
display = MAX7219(spi, cs)
display.fill(0)
display.show()

# -------------------------------------------------------
# FONT & DRAW
# -------------------------------------------------------
font_dict = {
    "C": b"\x3E\x41\x41\x41\x22",
    "D": b"\x7F\x41\x41\x22\x1C",
    "E": b"\x7F\x49\x49\x49\x41",
    "F": b"\x7F\x09\x09\x09\x01",
    "G": b"\x3E\x41\x49\x49\x7A",
}

def draw_letter(letter_str):
    display.fill(0)
    if letter_str in font_dict:
        columns = font_dict[letter_str]
        for col in range(5):
            byte_val = columns[col]
            for row in range(8):
                if (byte_val >> (7 - row)) & 1:
                    display.pixel(row, 6 - col, 1)
    display.show()

# -------------------------------------------------------
# BUTTONS
# -------------------------------------------------------
notes_map = [
    {"pin": 0, "note": 60, "letter": "C"},
    {"pin": 1, "note": 62, "letter": "D"},
    {"pin": 2, "note": 64, "letter": "E"},
    {"pin": 3, "note": 65, "letter": "F"},
    {"pin": 4, "note": 68, "letter": "G"},
]

buttons = []
initial_status = []
for item in notes_map:
    btn = machine.Pin(item["pin"], machine.Pin.IN, machine.Pin.PULL_UP)
    buttons.append(btn)
    initial_status.append(True)

# -------------------------------------------------------
# KY-036 TOUCH SENSOR (analog) -> drum hit
# Read on ADC pin 28. Untouched reads high (~65535); touching drops it.
# We trigger on the EDGE (not-touched -> touched) so one touch = one hit.
# Tune TOUCH_THRESHOLD by printing `raw` if it mis-triggers.
# -------------------------------------------------------
TOUCH_THRESHOLD = 65500
adc_pin = machine.ADC(machine.Pin(28))
touch_active = False

# -------------------------------------------------------
# ROTARY ENCODER (pitch bend)
# CLK=GP17, DT=GP18. Optional push-button SW=GP19 snaps bend to center.
# An interrupt catches every detent even if the main loop is busy; it only
# does quick integer math and sets a flag, the send happens in the loop.
# -------------------------------------------------------
clk = machine.Pin(17, machine.Pin.IN, machine.Pin.PULL_UP)
dt  = machine.Pin(18, machine.Pin.IN, machine.Pin.PULL_UP)
sw  = machine.Pin(19, machine.Pin.IN, machine.Pin.PULL_UP)   # optional; harmless if unwired

def encoder_turned(pin):
    # Fires once per detent (CLK falling edge). DT's level gives direction.
    global bend_value, bend_dirty
    if dt.value():
        bend_value -= BEND_STEP
    else:
        bend_value += BEND_STEP
    if bend_value < 0:
        bend_value = 0
    elif bend_value > 16383:
        bend_value = 16383
    bend_dirty = True

clk.irq(trigger=machine.Pin.IRQ_FALLING, handler=encoder_turned)

print("WiFi MIDI Instrument Active (pitch-bend encoder + touch drum)")

# -------------------------------------------------------
# MAIN LOOP
# -------------------------------------------------------
while True:
    # --- pitch bend: send whatever the encoder set since last loop ---
    if bend_dirty:
        bend_dirty = False
        pitch_bend(bend_value)

    # --- optional: press the encoder to snap pitch back to center ---
    if sw.value() == 0:
        bend_value = BEND_CENTER
        pitch_bend(bend_value)
        while sw.value() == 0:        # wait for release so it fires once
            time.sleep(0.01)

    # --- touch sensor -> drum hit (edge-triggered: one hit per touch) ---
    raw = adc_pin.read_u16()
    touched = raw < TOUCH_THRESHOLD
    if touched and not touch_active:
        print("Drum hit (touch)")
        drum_hit()
    touch_active = touched

    # --- buttons -> notes ---
    for i in range(len(buttons)):
        current_state = buttons[i].value()
        initial_state = initial_status[i]
        midi_note = notes_map[i]["note"]
        let = notes_map[i]["letter"]

        if not current_state and initial_state:
            print(f"Note ON: {midi_note}")
            note_on(midi_note)
            draw_letter(let)

        elif current_state and not initial_state:
            print(f"Note OFF: {midi_note}")
            note_off(midi_note)
            display.fill(0)
            display.show()

        initial_status[i] = current_state

    time.sleep(0.01)