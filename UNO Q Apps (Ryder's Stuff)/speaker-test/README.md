# 😀 Speaker Test

Connects to a Bluetooth speaker and plays a test melody through the `sound_generator` brick.

## Setup

1. Put your speaker in pairing mode.
2. On the board, find its MAC address:
   ```
   bluetoothctl
   > power on
   > scan on      # wait for it to show up, then Ctrl-C
   > devices
   ```
3. Edit `python/main.py` and set `BT_SPEAKER_MAC` to that address.
4. Run the app. It pairs/trusts/connects the speaker, then plays a short
   C-E-G-C chime through it.

If you hear nothing, check the ALSA device list the app logs at startup
(`aplay -L`) and adjust `ALSA_DEVICE` in `python/main.py` to match the
speaker's actual PCM name.
