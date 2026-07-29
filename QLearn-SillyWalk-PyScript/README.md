# Q-Learning Visualizer (SillyWalk-QLearn-V2)

A browser-based tool that trains a LEGO Education (SPIKE-style) robot with a
**Double Motor** unit to walk in a straight line using tabular **Q-learning**.
Everything — the UI, the training loop, the Bluetooth communication, and the
reinforcement-learning math — runs client-side in the browser via
[PyScript](https://pyscript.net/) (Python compiled to WebAssembly with
Pyodide). No server or install step is required beyond opening the page.

The robot reports its yaw (rotation drift) over Bluetooth Low Energy (BLE).
The page discretizes that yaw into a small number of "states" (e.g. Drifted
Left / Straight / Drifted Right), lets the agent choose an "action" (turn
LEFT, RIGHT, or optionally FORWARD), and updates a Q-table live using the
Bellman equation after every physical move — with the table, a Bellman
readout, and log messages updating in real time so you can watch the agent
learn.

## Files

| File | Purpose |
|---|---|
| `index.html` | The entire UI: layout, styles, and the JS glue that renders the Q-table, rewards table, yaw compass/strip-chart, and Bellman-update readout. Loads `main.py` as a PyScript module. |
| `main.py` | Application logic, in Python, run in-browser by PyScript. Handles BLE device connection, state discretization, the training/policy/step loops, and pushes updates back into the DOM via `window.*` JS functions defined in `index.html`. |
| `qlearn.py` | The `QTable` class: holds the Q-table and reward table (as NumPy arrays) and implements `bellman_update()` and epsilon-greedy `choose_action()`. |
| `ble.js` | A small `BLEDevice` wrapper around the browser's Web Bluetooth API (`navigator.bluetooth`) — connect, subscribe to notifications, write commands, disconnect. Imported into `main.py` as a JS module. |
| `pyscript.toml` | PyScript config: declares the `legoeducation` and `numpy` packages to install into the in-browser Python environment, ships `qlearn.py` as a local module, and registers `ble.js` as the `ble` JS module available to `main.py`. |

## How it fits together

1. **PyScript/Pyodide** loads a full Python interpreter into the browser and
   runs `main.py`, giving it access to `document`/`window` (via `pyscript`)
   and to NumPy and the `legoeducation` LEGO robotics package (both installed
   from `pyscript.toml`).
2. **`legoeducation`**'s device classes normally talk to hardware over a
   native BLE stack. Since there's no such stack in a browser, `main.py`
   patches the library's background worker (`_wasm_start_thread` /
   `_wasm_put_request`) to route its send/connect/disconnect requests through
   `ble.js`'s `BLEDevice` (Web Bluetooth) instead — this is what lets the
   unmodified `legoeducation.DoubleMotor` API work in-browser.
3. **The UI (`index.html`)** never talks to Python-specific state directly.
   It calls into Python only through `py-click` handlers (e.g.
   `py-click="start_training"`) and one `manual-action` custom event (used
   for dynamically-created buttons, since `py-click` isn't reliably picked up
   on elements injected after page load). Python calls back into the DOM
   through a set of `window.*` functions defined in `index.html`'s `<script>`
   block (`updateCell`, `pulseCell`, `highlightStateRow`,
   `updateBellmanReadout`, `updateYawGauge`, etc.) to keep the visuals in
   sync with the Q-table.

## Requirements

- A browser with **Web Bluetooth** support (Chrome or Edge on desktop;
  Bluefy on iOS). Safari and Firefox do not support Web Bluetooth.
- The page must be served over `http://` or `https://` (or `localhost`) —
  opening `index.html` directly as a `file://` URL will not work, since both
  PyScript and Web Bluetooth require a proper origin.
- A LEGO Education robot with a **Double Motor** and **IMU** peripheral,
  advertising the BLE service UUID hardcoded in `main.py`
  (`0000fd02-0000-1000-8000-00805f9b34fb`).

## Running it

Serve the folder with any static file server and open it in a supported
browser, for example:

```bash
python3 -m http.server 8000
```

then visit `http://localhost:8000/index.html`.

## Using the page

- **Hardware Setup (sidebar)** — click **Connect** to pair with the Double
  Motor over Web Bluetooth (a native browser device picker will appear).
  **Show Yaw** opens a live modal with a numeric readout, a compass needle,
  and a 30-second history strip chart of the robot's yaw. **Reset Yaw**
  zeroes the IMU's yaw axis.
- **Start Training / Stop / Step Once / Run Learned Policy** — run a full
  multi-episode training loop, stop whatever loop is active, advance exactly
  one state/action/Bellman-update transition at a time (useful for teaching),
  or run the greedy (`eps=0`) policy learned so far.
- **Settings panel** — collapsible subsections for:
  - **Turn Settings**: how many degrees the motors turn per step.
  - **Training Settings**: steps per episode and number of episodes (yaw
    resets to 0 at the start of each episode; epsilon decays across all
    steps of all episodes).
  - **Q-Learning Parameters**: alpha (learning rate) and gamma (discount
    factor).
  - **Rewards Configuration**: a per-(state, action) reward slider, mirrring
    the Q-table's layout.
  - **Apply Settings** commits all of the above into the running `QTable`;
    **Reset Q-Table** zeroes the learned values back out.
- **States / Actions dropdowns** — switch between 3 or 5 discretized yaw
  states, and between a 2-action (LEFT/RIGHT) or 3-action
  (LEFT/RIGHT/FORWARD) space. Changing either rebuilds the Q-table and
  rewards table and resets learned values (a running loop must be stopped
  first).
- **Q-Table** — one row per state, one column per action, color-coded green
  (positive) / red (negative) by Q-value magnitude. Each action column header
  has a small ▶ button to physically execute that action once without
  touching the Q-table. Below the table, the **Bellman update** box shows the
  literal numbers from the most recent update.
- **Logs** — a running terminal-style log of every connection event,
  training step, and error.
