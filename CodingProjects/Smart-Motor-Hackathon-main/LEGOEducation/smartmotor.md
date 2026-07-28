# LEGO® Education Python API

![LEGO Education logo](./img/LEGOEducation.png)

1. [Introduction and Installation](./README.md)
2. [Connect and Run](./connect.md)
3. [Single Motor](./singlemotor.md)
4. [Double Motor](./doublemotor.md)
5. [Color Sensor](./colorsensor.md)
6. [Controller](./controller.md)
7. [Combine Single Motor and Color Sensor](./combine1.md)
8. [Combine Double Motor and Controller](./combine2.md)
9. [Constants](./constants.md)
10. **Smart Motor (Teachable)**

---
# Smart Motor (Teachable)

The Smart Motor example is a graphical program that lets you *teach* a motor how
to react to a sensor, then *run* it — similar in spirit to a teachable "smart
motor". It connects a **sensor** (Color Sensor or Controller) to a motor
(**Single Motor** or **Double Motor**) through a simple nearest-neighbor model.

There are two versions with identical behavior:

- **Desktop app** — [smart_motor.py](./examples/smart_motor.py) (a Tkinter window;
  needs `matplotlib`).
- **Web app** — [smart_motor_web.py](./examples/smart_motor_web.py) (runs in your
  browser via a small local server; see [Web version](#web-version) below).

## How it works

- **Training mode** — You record data points. Each point pairs one *sensor
  reading* with one *motor position*, drawn on a live graph where the x-axis is
  the sensor reading and the y-axis is the motor position.
- **Run mode** — The live sensor reading is read continuously. The program finds
  the recorded point whose sensor reading is *closest* to the live reading and
  moves the motor to that point's position. You watch the sensor input and the
  resulting motor movement on the same graph.

A **Double Motor** records both the left and right positions for each point (two
series on the graph), and run mode drives both motors to the nearest point.

## Requirements

```
pip install matplotlib
pip install legoeducation   # only needed for the LEGO Hardware backend
```

`matplotlib` provides the graph and is part of Python's wider ecosystem;
`tkinter` (the window toolkit) ships with the standard Python installer.

## Running

```
python smart_motor.py
```

### Backends

- **Simulated** — No hardware required. Sliders stand in for the sensor and, in
  training mode, the motor position. Use this to explore the whole interface
  first.
- **LEGO Hardware** — Connects to a real motor plus a sensor. Choose the
  **Motor** (Single or Double) and the **Sensor** (Color Sensor or Controller),
  then press **Connect**.

  The motor and the sensor are *separate* Bluetooth devices. By default the
  program connects to the **first motor and first sensor it finds** (the
  `legoeducation` default), so you usually don't need to enter anything. If you
  have more than one of a kind broadcasting, tick **"Filter by connection card"**
  and enter the card color + serial for the motor and the sensor *separately*
  (leaving a serial blank matches on color only). See
  [Connect and Run](./connect.md).

> **Connection tip:** an error like *"Could not find device matching Card color
> …, Card serial …"* means no broadcasting device matched that card. Make sure
> the hardware is charged, powered on, and broadcasting, and either leave
> filtering **off** (connect to first found) or enter the *correct* card values
> printed for each device.

## Typical workflow

1. Pick a backend and (for hardware) press **Connect**.
2. Tick which sensor **feature(s)** to map from. You can select more than one —
   for example, tick **both** the Controller's `leftPercent` and `rightPercent`
   so both levers control the motor at the same time. The nearest-neighbor match
   is computed over all ticked features together (Euclidean distance). Use the
   **Graph x-axis** picker to choose which ticked feature the graph plots.
   (Changing the ticked set clears existing points, since they were recorded with
   a different set of inputs.)
3. In **Training** mode, move the motor to a position, set/observe the sensor
   reading, and press **Record point**. Repeat to build up the graph.
4. Press **Switch to RUN mode**. Change the sensor input and watch the motor
   move to the position of the nearest recorded point.
5. Use **Save…/Load…** to keep a set of trained points as a JSON file.

## Sensor features

| Sensor      | Available features                                        |
| ----------- | --------------------------------------------------------- |
| Color Sensor| `reflection`, `hue`, `color`, `value`, `saturation`       |
| Controller  | `leftPercent`, `rightPercent`, `leftAngle`, `rightAngle`  |

## Multiple sensor inputs

Each recorded point stores a *vector* of the ticked sensor readings paired with
the motor position(s). In run mode the program measures the distance from the
live readings to every recorded point (across all ticked features at once) and
drives the motor to the closest one. This is how both Controller levers can act
on the motor simultaneously — a point recorded with the left lever pushed and a
point recorded with the right lever pushed are simply two neighbors in the same
space. Because features are compared with plain Euclidean distance, mixing
features with very different ranges (e.g. `hue` 0–360 with `reflection` 0–100)
lets the larger-range feature dominate; prefer features on similar scales.

Motor positions are read from each motor's relative `position` and driven with
`motor_run_to_relative_position`, using the connection point as the zero
reference (reset automatically on connect). A Double Motor uses `MOTOR_LEFT` and
`MOTOR_RIGHT` independently.

## Web version

[smart_motor_web.py](./examples/smart_motor_web.py) provides the same interface
in a **web browser**. It starts a small local web server (Python standard library
only — no extra packages beyond `legoeducation`) and serves a page you control
from your browser. The graph is drawn on an HTML canvas, so nothing is downloaded
from the internet.

```
python smart_motor_web.py
```

### Mapping a Double Motor

With a **Double Motor** the web app adds a **Mapping** choice:

- **Combined** (default) — all selected sensor readings drive both motors
  together as one nearest-neighbor model (the behavior described above).
- **Independent (one input per motor)** — each motor gets its **own** input
  feature and its **own** training points. For a Controller this defaults to the
  left motor being driven by `leftPercent` and the right motor by `rightPercent`,
  but you can assign **any lever to any motor** (including swapping them). Each
  motor has its own **Record** / **Clear** buttons, so recording the left side
  stores only the left motor's position — it never captures the right motor.

  In run mode both inputs act at once, like a normal game controller: pressing
  the left lever turns its motor, pressing the right lever turns the other motor,
  and pressing both turns both — each side follows its own lever independently.

Then open the printed address (default <http://127.0.0.1:8000>). Options:

- `--port N` — use a different port.
- `--host 0.0.0.0` — allow other devices on your network to reach it.
- `--no-browser` — don't open a browser automatically.

**Why a local server instead of a plain website?** The LEGO Education hardware
communicates over Bluetooth Low Energy using the `legoeducation` Python package.
A web page on its own cannot speak that protocol, so a hosted website could not
move a real motor. This program keeps the Bluetooth work in Python (which *can*
access Bluetooth) and puts only the interface in the browser. Everything runs on
your own machine.

---

**Back to:** [Introduction and Installation](./README.md)
