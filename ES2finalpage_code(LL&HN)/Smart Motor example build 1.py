from machine import Pin, SoftI2C, ADC
import network
import socket
import time
import ssd1306
# ---------------- CONFIG: fill these in ----------------
WIFI_SSID = "LocalNetwork"
WIFI_PASSWORD = "musichackathon"
PI_IP = "10.42.0.1"      # Raspberry Pi's IP address
PI_PORT = 5010              # UDP port your Pi-side script listens on
MIDI_CHANNEL = 0            # 0 = MIDI channel 1 - give each board a different value


INSTRUMENT_PROGRAM = 0
VOL_UP_PIN = 8       # the board's original "up" nav switch
VOL_DOWN_PIN = 10     # the board's original "down" nav switch
# ---------------------------------------------------------
# ---- Wi-Fi ----
wlan = network.WLAN(network.STA_IF)
wlan.active(True)
wlan.connect(WIFI_SSID, WIFI_PASSWORD)
print("Connecting to WiFi...")
while not wlan.isconnected():
    time.sleep(0.5)
print("Connected:", wlan.ifconfig())
# ---- UDP ----
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
def send_midi(status, data1, data2=0):
    try:
        sock.sendto(bytes([status, data1 & 0x7F, data2 & 0x7F]), (PI_IP, PI_PORT))
    except OSError as e:
        print("MIDI send failed:", e)
        
def note_on(note, velocity=100):
    send_midi(0x90 | MIDI_CHANNEL, note, velocity)
def note_off(note):
    send_midi(0x80 | MIDI_CHANNEL, note, 0)
def program_change(program):
    send_midi(0xC0 | MIDI_CHANNEL, program)
def control_change(cc, value):
    send_midi(0xB0 | MIDI_CHANNEL, cc, value)
# ---- OLED display (reusing the board's existing ssd1306.py) ----
i2c = SoftI2C(scl=Pin(7), sda=Pin(6))
display = ssd1306.SSD1306_I2C(128, 64, i2c)
_last_shown = None
def show(line1, line2=""):
    global _last_shown
    key = (line1, line2)
    if key == _last_shown:
        return  # skip redundant I2C writes
    _last_shown = key
    display.fill(0)
    display.text(line1, 0, 20, 1)
    display.text(line2, 0, 35, 1)
    display.show()
    
# ---- Note button divider ----
adc_buttons = ADC(Pin(5))
adc_buttons.atten(ADC.ATTN_11DB)

THRESHOLDS = [186, 676, 1393, 2015, 2519, 2936, 3094, 3614]
def read_button():
    val = adc_buttons.read()
    if val > THRESHOLDS[7]:
        return None
    elif val > THRESHOLDS[6]:
        return 6
    elif val > THRESHOLDS[5]:
        return 5
    elif val > THRESHOLDS[4]:
        return 4
    elif val > THRESHOLDS[3]:
        return 3
    elif val > THRESHOLDS[2]:
        return 2
    elif val > THRESHOLDS[1]:
        return 1
    elif val > THRESHOLDS[0]:
        return 0
    else:
        return None
# ---- mode toggle: reuse existing "select" switch (GPIO9) ----
mode_button = Pin(9, Pin.IN)
mode = 0
last_mode_btn = mode_button.value()
# ---- volume up/down (side buttons - pins TBD) ----
vol_up_btn = Pin(VOL_UP_PIN, Pin.IN, Pin.PULL_UP) if VOL_UP_PIN is not None else None
vol_down_btn = Pin(VOL_DOWN_PIN, Pin.IN, Pin.PULL_UP) if VOL_DOWN_PIN is not None else None
last_vol_up = 1
last_vol_down = 1
volume = 100
BASE_NOTES = [60, 62, 64, 65, 67, 69, 71]  # C4 D4 E4 F4 G4 A4 B4
NOTE_NAMES = ["C", "D", "E", "F", "G", "A", "B"]
current_note = None
program_change(INSTRUMENT_PROGRAM)  # set this board's instrument on the Pi
print("Ready.")
show("Ready", "mode:0")
while True:
    # --- mode toggle (select switch), crude debounce ---
    btn_state = mode_button.value()
    if btn_state == 0 and last_mode_btn == 1:
        mode = 1 - mode
        time.sleep_ms(150)
    last_mode_btn = btn_state
    # --- volume buttons (only active once pins are filled in) ---
    if vol_up_btn:
        v = vol_up_btn.value()
        if v == 0 and last_vol_up == 1:
            volume = min(127, volume + 8)
            control_change(7, volume)
            time.sleep_ms(120)
        last_vol_up = v
    if vol_down_btn:
        v = vol_down_btn.value()
        if v == 0 and last_vol_down == 1:
            volume = max(0, volume - 8)
            control_change(7, volume)
            time.sleep_ms(120)
        last_vol_down = v
    # --- which note button is pressed? ---
    idx = read_button()
    note = (BASE_NOTES[idx] + (12 if mode else 0)) if idx is not None else None
    if note != current_note:
        print("DEBUG note change:", current_note, "->", note, " (idx=", idx, ")")
        if current_note is not None:
            note_off(current_note)
        if note is not None:
            note_on(note)
        current_note = note
    # --- update display only when something changed ---
    note_label = "{}{}".format(NOTE_NAMES[idx], 4 + mode) if idx is not None else "--"
    show(
        "Note:{} mode:{}".format(note_label, mode),
        "Vol:{}".format(volume),
    )
    time.sleep_ms(10)
