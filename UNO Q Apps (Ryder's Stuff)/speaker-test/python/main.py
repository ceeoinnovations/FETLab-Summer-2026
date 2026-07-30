import subprocess
import time

from arduino.app_bricks.sound_generator import SoundGenerator
from arduino.app_peripherals.speaker import Speaker
from arduino.app_utils import App, Logger

logger = Logger(__name__)

# --- Configuration ------------------------------------------------------
# Set this to your Bluetooth speaker's MAC address, e.g. "AA:BB:CC:DD:EE:FF".
# Find it on the board with:
#   bluetoothctl
#   > power on
#   > scan on        (wait for your speaker to appear, then Ctrl-C)
#   > devices
BT_SPEAKER_MAC = "F4:2B:7D:5B:46:5B"

# ALSA playback device to use once the speaker is connected. "pipewire"
# routes to whichever sink PipeWire currently considers default, which is
# normally the Bluetooth speaker right after it connects. If audio doesn't
# come out, run `aplay -L` on the board (this app logs it at startup too)
# and set this to the exact PCM name shown there for the speaker.
ALSA_DEVICE = "pipewire"


def run_bluetoothctl(*args, timeout=15):
    """Run a bluetoothctl subcommand, return (success, stdout)."""
    try:
        result = subprocess.run(
            ["bluetoothctl", *args],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode == 0, result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return False, str(e)


def connect_bluetooth_speaker(mac: str) -> bool:
    """Make sure the adapter is on and the speaker is paired, trusted and connected."""
    run_bluetoothctl("power", "on")

    _, info = run_bluetoothctl("info", mac)
    if "Paired: yes" not in info:
        logger.info(f"Pairing with {mac} ...")
        run_bluetoothctl("pair", mac)
        run_bluetoothctl("trust", mac)

    _, info = run_bluetoothctl("info", mac)
    if "Connected: yes" not in info:
        logger.info(f"Connecting to {mac} ...")
        run_bluetoothctl("connect", mac)
        time.sleep(2)  # give PipeWire/WirePlumber time to expose the A2DP sink

    _, info = run_bluetoothctl("info", mac)
    connected = "Connected: yes" in info
    logger.info(f"Bluetooth speaker {mac} connected: {connected}")
    return connected


def log_available_audio_devices():
    result = subprocess.run(["aplay", "-L"], capture_output=True, text=True)
    logger.info(f"Available ALSA playback devices:\n{result.stdout}")


def play_test_sound(device):
    speaker = Speaker(device=device)
    gen = SoundGenerator(output_device=speaker, wave_form="sine")
    gen.start()
    gen.set_master_volume(0.8)

    logger.info(f"Playing test sound on device={device!r} ...")
    for note in ["C4", "E4", "G4", "C5"]:
        gen.play(note, note_duration=1 / 4, block=True)
    logger.info("Done.")


def loop():
    time.sleep(10)


if not BT_SPEAKER_MAC:
    logger.warning(
        "BT_SPEAKER_MAC is not set in python/main.py — edit it with your "
        "speaker's MAC address. Playing the test sound on the default "
        "speaker for now."
    )
    play_test_sound(Speaker.USB_SPEAKER_1)
else:
    log_available_audio_devices()
    if connect_bluetooth_speaker(BT_SPEAKER_MAC):
        play_test_sound(ALSA_DEVICE)
    else:
        logger.error(f"Could not connect to Bluetooth speaker {BT_SPEAKER_MAC}")

App.run(user_loop=loop)
