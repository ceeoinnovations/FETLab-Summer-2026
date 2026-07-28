# simple-test

Joystick-controlled tank drive for a LEGO Education double motor, using the [SimpleLE](https://github.com/chrisbuerginrogers/SimpleLE) wrapper library.

## What it does

`main.py` connects to a LEGO Education **double motor** and a LEGO **controller (joystick)** over Bluetooth, then runs a continuous loop that reads both joystick positions and feeds them directly into a tank-drive command:

- **Left joystick** → left motor speed
- **Right joystick** → right motor speed

Pushing a joystick forward drives that side's motor forward; pulling it back reverses it. This lets you steer like a tank — push both forward to go straight, push one and pull the other to spin in place.

The loop runs at roughly 20 Hz (50 ms per cycle). When you press `Ctrl+C`, the motors stop cleanly before the program exits.

## Files

| File | Purpose |
|---|---|
| `main.py` | Main script — connects devices and runs the drive loop |
| `lelib.py` | SimpleLE wrapper around the `legoeducation` package |
| `requirements.txt` | Python dependencies |

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Edit `main.py` and set `SERIAL` to your Bluetooth card's serial number:
   ```python
   SERIAL = 1128  # ← change this
   ```

3. Run:
   ```bash
   python main.py
   ```

## Controls

| Joystick | Effect |
|---|---|
| Left stick up | Left motor forward |
| Left stick down | Left motor backward |
| Right stick up | Right motor forward |
| Right stick down | Right motor backward |
| Both sticks centered | Motors stop |
