"""
Smart Motor — a teachable sensor-to-motor interface for LEGO(R) Education hardware.

This program connects a *sensor* (Color Sensor or Controller) to a motor
(Single Motor or Double Motor) and lets you teach the motor how to react to the
sensor, then run it, all from a graphical interface.

How it works
------------
* TRAINING MODE
    You record data points. Each point pairs the current *sensor reading(s)* with
    the *motor position(s)*. The points are drawn on a live graph:
        x-axis = a sensor reading, y-axis = motor position.
    A Double Motor records both the left and right positions (two series).

* RUN MODE
    The live sensor reading(s) are read continuously. The program finds the
    recorded point whose readings are *closest* to the live readings (nearest-
    neighbor) and moves the motor to that point's position. You can watch the
    sensor input and the resulting motor movement on the same graph.

Multiple sensor inputs
----------------------
You can select more than one sensor feature at once (checkboxes). For example,
select BOTH the Controller's `leftPercent` and `rightPercent` so both levers
control the motor simultaneously: the nearest-neighbor match is computed over all
selected features together (Euclidean distance). The "Graph x-axis" picker chooses
which selected feature is drawn on the graph.

Backends
--------
* "Simulated" backend: no hardware needed. Sliders stand in for the sensor(s) and
    (during training) the motor, so you can explore the whole interface first.
* "LEGO Hardware" backend: uses the `legoeducation` package to talk to a real
    Single/Double Motor plus a Color Sensor or Controller over Bluetooth.

Connecting to hardware
----------------------
The motor and the sensor are *separate* Bluetooth devices, each with its own
Connection Card. By default this program connects to the *first* motor and the
*first* sensor it finds. If you have more than one of a kind broadcasting, tick
"Filter by connection card" and enter the card color + serial for each separately.

Requirements
------------
    pip install matplotlib
    pip install legoeducation      # only needed for the LEGO Hardware backend

Then run:
    python smart_motor.py
"""

import json
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# --- Optional dependency: matplotlib (for the graph) -------------------------
try:
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    HAS_MPL = True
except ImportError:  # pragma: no cover
    HAS_MPL = False

# --- Optional dependency: legoeducation (for real hardware) ------------------
try:
    import legoeducation as le
    HAS_LE = True
except ImportError:
    le = None
    HAS_LE = False


# =============================================================================
# Model: nearest-neighbor mapping from a sensor reading vector to positions
# =============================================================================
class NearestNeighborModel:
    """Stores (sensor_vector, [positions...]) points; predicts by closest match.

    * sensor_vector is a list with one entry per selected sensor feature.
    * positions is a list: one entry for a Single Motor, two (left, right) for a
      Double Motor.

    All points share the same set of feature names (`features`); changing the
    feature set requires clearing the points first, so every stored vector lines
    up with every other.
    """

    def __init__(self):
        self._points = []          # list of [sensor_vector, positions]
        self._features = []        # feature names for each sensor_vector entry
        self._lock = threading.Lock()

    @property
    def features(self):
        with self._lock:
            return list(self._features)

    def set_features(self, features):
        """Set the sensor feature names. Returns False if points already exist
        with a *different* feature set (caller should clear first)."""
        features = list(features)
        with self._lock:
            if features == self._features:
                return True
            if self._points:
                return False
            self._features = features
            return True

    def add_point(self, sensor_vector, positions):
        vec = [float(x) for x in sensor_vector]
        pos = [float(x) for x in positions]
        with self._lock:
            self._points.append([vec, pos])

    def clear(self):
        with self._lock:
            self._points = []

    def points(self):
        with self._lock:
            return [[list(v), list(p)] for v, p in self._points]

    def __len__(self):
        with self._lock:
            return len(self._points)

    def predict(self, sensor_vector):
        """Return the positions of the point closest to sensor_vector
        (Euclidean distance). Returns None when there are no recorded points."""
        with self._lock:
            if not self._points:
                return None

            def dist2(point):
                return sum((a - b) ** 2 for a, b in zip(point[0], sensor_vector))

            best = min(self._points, key=dist2)
            return list(best[1])

    def to_json(self):
        return json.dumps({"features": self.features, "points": self.points()}, indent=2)

    def load_json(self, text):
        data = json.loads(text)
        pts = data.get("points", [])
        features = data.get("features", [])
        parsed = []
        for entry in pts:
            s, p = entry
            # Back-compat: older files stored a scalar sensor value and/or a
            # scalar motor position.
            if isinstance(s, (int, float)):
                s = [s]
            if isinstance(p, (int, float)):
                p = [p]
            parsed.append([[float(x) for x in s], [float(x) for x in p]])
        if not features and parsed:
            features = [f"feature{i}" for i in range(len(parsed[0][0]))]
        with self._lock:
            self._features = list(features)
            self._points = parsed


# =============================================================================
# Backends: a common interface for reading a sensor and driving a motor
# =============================================================================
class Backend:
    """Abstract backend.

    Reports sensor readings and a list of motor positions (degrees). The motor
    position list has one entry per motor axis (see `motor_labels`).
    """

    name = "base"
    sensor_features = ["value"]      # feature names selectable in the UI
    motor_labels = ["motor"]         # one label per motor axis

    def connect(self, **kwargs):
        raise NotImplementedError

    def disconnect(self):
        pass

    @property
    def connected(self):
        return False

    def read_sensor_vector(self, features):
        """Return a list of current readings, one per name in `features`."""
        raise NotImplementedError

    def read_motor_position(self):
        """Return a list of current motor positions (one per axis)."""
        raise NotImplementedError

    def move_motor_to(self, positions):
        """Command each motor axis toward positions[i] (degrees)."""
        raise NotImplementedError


class SimulatedBackend(Backend):
    """Hardware-free backend driven by UI sliders. Great for trying the interface.

    Exposes two sensor features (valueX, valueY) so the multi-input behavior can
    be explored without a real Controller.
    """

    name = "Simulated"
    sensor_features = ["valueX", "valueY"]
    motor_labels = ["motor"]

    def __init__(self):
        self._connected = False
        self._sensor = {"valueX": 50.0, "valueY": 50.0}
        self._motor_pos = 0.0
        self._motor_target = 0.0
        self._lock = threading.Lock()

    def connect(self, **kwargs):
        self._connected = True
        return True

    def disconnect(self):
        self._connected = False

    @property
    def connected(self):
        return self._connected

    # -- UI hooks (called from the GUI thread) --
    def set_sim_sensor(self, feature, value):
        with self._lock:
            self._sensor[feature] = float(value)

    def set_manual_motor(self, value):
        """During training the motor position is set by hand (a slider here)."""
        with self._lock:
            self._motor_pos = float(value)
            self._motor_target = float(value)

    # -- Backend interface --
    def read_sensor_vector(self, features):
        with self._lock:
            return [self._sensor.get(f, 0.0) for f in features]

    def read_motor_position(self):
        with self._lock:
            diff = self._motor_target - self._motor_pos
            self._motor_pos += diff * 0.35
            if abs(diff) < 0.5:
                self._motor_pos = self._motor_target
            return [self._motor_pos]

    def move_motor_to(self, positions):
        with self._lock:
            self._motor_target = float(positions[0])


class LegoBackend(Backend):
    """Real LEGO Education hardware: a motor plus a Color Sensor or Controller."""

    name = "LEGO Hardware"

    COLOR_FEATURES = ["reflection", "hue", "color", "value", "saturation"]
    CONTROLLER_FEATURES = ["leftPercent", "rightPercent", "leftAngle", "rightAngle"]

    def __init__(self, sensor_kind="color", motor_kind="single"):
        self.sensor_kind = sensor_kind          # "color" | "controller"
        self.motor_kind = motor_kind            # "single" | "double"
        self.sensor_features = (
            self.COLOR_FEATURES if sensor_kind == "color" else self.CONTROLLER_FEATURES
        )
        self.motor_labels = ["motor"] if motor_kind == "single" else ["left", "right"]
        self.motor = None
        self.sensor = None

    def connect(self, motor_card=None, sensor_card=None):
        """Connect the motor and the sensor.

        motor_card / sensor_card are dicts like {"card_color": ..., "card_serial": ...}
        or None to connect to the first device found.
        """
        if not HAS_LE:
            raise RuntimeError("The 'legoeducation' package is not installed.")

        self.motor = le.SingleMotor() if self.motor_kind == "single" else le.DoubleMotor()
        self.motor.connect(**(motor_card or {}))

        self.sensor = le.ColorSensor() if self.sensor_kind == "color" else le.Controller()
        self.sensor.connect(**(sensor_card or {}))

        if not self.motor.connected:
            raise RuntimeError("Could not connect to the motor.")
        if not self.sensor.connected:
            raise RuntimeError("Could not connect to the sensor.")

        # Use the current motor position(s) as the zero reference.
        if self.motor_kind == "single":
            self.motor.motor_reset_relative_position()
        else:
            self.motor.motor_reset_relative_position(motor=le.MOTOR_LEFT)
            self.motor.motor_reset_relative_position(motor=le.MOTOR_RIGHT)
        return True

    def disconnect(self):
        try:
            if self.motor:
                self.motor.motor_stop()
                self.motor.disconnect()
        finally:
            if self.sensor:
                self.sensor.disconnect()

    @property
    def connected(self):
        return bool(self.motor and self.sensor
                    and self.motor.connected and self.sensor.connected)

    def read_sensor_vector(self, features):
        return [float(getattr(self.sensor.sensor, f)) for f in features]

    def read_motor_position(self):
        if self.motor_kind == "single":
            return [float(self.motor.motor.position)]
        return [float(self.motor.motor[le.MOTOR_LEFT].position),
                float(self.motor.motor[le.MOTOR_RIGHT].position)]

    def move_motor_to(self, positions):
        if self.motor_kind == "single":
            self.motor.motor_run_to_relative_position(
                int(round(positions[0])), blocking=False)
        else:
            self.motor.motor_run_to_relative_position(
                int(round(positions[0])), motor=le.MOTOR_LEFT, blocking=False)
            self.motor.motor_run_to_relative_position(
                int(round(positions[1])), motor=le.MOTOR_RIGHT, blocking=False)


# =============================================================================
# The application (GUI + background control loop)
# =============================================================================
MODE_TRAINING = "TRAINING"
MODE_RUN = "RUN"
SERIES_COLORS = ["#1f77b4", "#ff7f0e"]

# Which features are checked by default for each backend.
DEFAULT_FEATURES = {
    "sim": ["valueX"],
    "color": ["reflection"],
    "controller": ["leftPercent", "rightPercent"],
}


class SmartMotorApp:
    POLL_MS = 100  # GUI refresh interval

    def __init__(self, root):
        self.root = root
        self.root.title("Smart Motor — teachable sensor-to-motor")

        self.model = NearestNeighborModel()
        self.backend = SimulatedBackend()
        self.mode = MODE_TRAINING

        # Feature-selection state.
        self.feature_vars = {}           # feature name -> BooleanVar
        self._committed_features = []    # last accepted selection (GUI thread)
        # Plain-list mirror of the selection, safe to read from the worker
        # thread (Tkinter variables must not be touched off the GUI thread).
        self._active_features = []

        # Latest readings shared between the control thread and the GUI.
        self._latest_vec = [0.0]
        self._latest_features = []
        self._latest_pos = [0.0]
        self._latest_target = None
        self._state_lock = threading.Lock()

        self._worker = None
        self._worker_stop = threading.Event()
        self.series = []  # per-axis matplotlib artists

        self._build_ui()
        self._rebuild_backend()          # sets features, sim inputs, plot
        self._update_mode_ui()
        self.root.after(self.POLL_MS, self._tick)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ----------------------------------------------------------------- UI ----
    def _build_ui(self):
        main = ttk.Frame(self.root, padding=8)
        main.pack(fill="both", expand=True)

        controls = ttk.Frame(main)
        controls.pack(side="left", fill="y", padx=(0, 10))

        # ---- Hardware group ----
        conn = ttk.LabelFrame(controls, text="Hardware", padding=8)
        conn.pack(fill="x", pady=4)
        conn.columnconfigure(1, weight=1)

        ttk.Label(conn, text="Backend:").grid(row=0, column=0, sticky="w")
        self.backend_var = tk.StringVar(value=SimulatedBackend.name)
        backend_choices = [SimulatedBackend.name] + (["LEGO Hardware"] if HAS_LE else [])
        self.backend_combo = ttk.Combobox(
            conn, textvariable=self.backend_var, values=backend_choices,
            state="readonly", width=18)
        self.backend_combo.grid(row=0, column=1, sticky="ew", pady=2)
        self.backend_combo.bind("<<ComboboxSelected>>", lambda e: self._rebuild_backend())

        ttk.Label(conn, text="Motor:").grid(row=1, column=0, sticky="w")
        self.motor_var = tk.StringVar(value="Single Motor")
        self.motor_combo = ttk.Combobox(
            conn, textvariable=self.motor_var,
            values=["Single Motor", "Double Motor"], state="readonly", width=18)
        self.motor_combo.grid(row=1, column=1, sticky="ew", pady=2)
        self.motor_combo.bind("<<ComboboxSelected>>", lambda e: self._rebuild_backend())

        ttk.Label(conn, text="Sensor:").grid(row=2, column=0, sticky="w")
        self.sensor_kind_var = tk.StringVar(value="Color Sensor")
        self.sensor_kind_combo = ttk.Combobox(
            conn, textvariable=self.sensor_kind_var,
            values=["Color Sensor", "Controller"], state="readonly", width=18)
        self.sensor_kind_combo.grid(row=2, column=1, sticky="ew", pady=2)
        self.sensor_kind_combo.bind("<<ComboboxSelected>>", lambda e: self._rebuild_backend())

        # Card filtering (optional)
        self.filter_var = tk.BooleanVar(value=False)
        self.filter_check = ttk.Checkbutton(
            conn, text="Filter by connection card", variable=self.filter_var,
            command=self._update_hw_widgets)
        self.filter_check.grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 0))

        color_names = ["GREEN", "BLUE", "RED", "ORANGE", "YELLOW",
                       "AZURE", "PURPLE", "MAGENTA"]

        ttk.Label(conn, text="Motor card:").grid(row=4, column=0, sticky="w")
        card_row_m = ttk.Frame(conn)
        card_row_m.grid(row=4, column=1, sticky="ew", pady=2)
        self.motor_color_var = tk.StringVar(value="AZURE")
        self.motor_color_combo = ttk.Combobox(
            card_row_m, textvariable=self.motor_color_var, values=color_names,
            state="readonly", width=8)
        self.motor_color_combo.pack(side="left")
        self.motor_serial_var = tk.StringVar(value="")
        self.motor_serial_entry = ttk.Entry(
            card_row_m, textvariable=self.motor_serial_var, width=8)
        self.motor_serial_entry.pack(side="left", padx=(4, 0))

        ttk.Label(conn, text="Sensor card:").grid(row=5, column=0, sticky="w")
        card_row_s = ttk.Frame(conn)
        card_row_s.grid(row=5, column=1, sticky="ew", pady=2)
        self.sensor_color_var = tk.StringVar(value="AZURE")
        self.sensor_color_combo = ttk.Combobox(
            card_row_s, textvariable=self.sensor_color_var, values=color_names,
            state="readonly", width=8)
        self.sensor_color_combo.pack(side="left")
        self.sensor_serial_var = tk.StringVar(value="")
        self.sensor_serial_entry = ttk.Entry(
            card_row_s, textvariable=self.sensor_serial_var, width=8)
        self.sensor_serial_entry.pack(side="left", padx=(4, 0))

        self.connect_btn = ttk.Button(conn, text="Connect", command=self._on_connect)
        self.connect_btn.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        self.conn_status = ttk.Label(conn, text="Not connected", foreground="#a00")
        self.conn_status.grid(row=7, column=0, columnspan=2, sticky="w", pady=(4, 0))

        # ---- Sensor inputs group (multi-select) ----
        feat = ttk.LabelFrame(controls, text="Sensor inputs", padding=8)
        feat.pack(fill="x", pady=4)
        ttk.Label(feat, text="Tick every reading that should drive the motor:").pack(
            anchor="w")
        self.feature_check_frame = ttk.Frame(feat)
        self.feature_check_frame.pack(fill="x", pady=(2, 4))
        axis_row = ttk.Frame(feat)
        axis_row.pack(fill="x")
        ttk.Label(axis_row, text="Graph x-axis:").pack(side="left")
        self.display_feature_var = tk.StringVar()
        self.display_combo = ttk.Combobox(
            axis_row, textvariable=self.display_feature_var, state="readonly", width=14)
        self.display_combo.pack(side="left", padx=(4, 0))
        self.display_combo.bind("<<ComboboxSelected>>", lambda e: (self._redraw_points()))

        # ---- Simulation inputs ----
        self.sim_frame = ttk.LabelFrame(controls, text="Simulation inputs", padding=8)
        self.sim_frame.pack(fill="x", pady=4)
        self.sim_sensor_frame = ttk.Frame(self.sim_frame)
        self.sim_sensor_frame.pack(fill="x")
        self.sim_sensor_vars = {}  # feature -> DoubleVar
        self.sim_motor_label = ttk.Label(self.sim_frame, text="Motor position (training)")
        self.sim_motor_label.pack(anchor="w", pady=(6, 0))
        self.sim_motor_var = tk.DoubleVar(value=0)
        self.sim_motor_scale = ttk.Scale(
            self.sim_frame, from_=-180, to=180, variable=self.sim_motor_var,
            command=self._on_sim_motor)
        self.sim_motor_scale.pack(fill="x")

        # ---- Mode group ----
        mode_box = ttk.LabelFrame(controls, text="Mode", padding=8)
        mode_box.pack(fill="x", pady=4)
        self.mode_btn = ttk.Button(mode_box, text="Switch to RUN mode",
                                   command=self._toggle_mode)
        self.mode_btn.pack(fill="x")
        self.mode_label = ttk.Label(mode_box, text="Current: TRAINING",
                                    font=("TkDefaultFont", 10, "bold"))
        self.mode_label.pack(anchor="w", pady=(4, 0))

        # ---- Training actions ----
        self.train_box = ttk.LabelFrame(controls, text="Training", padding=8)
        self.train_box.pack(fill="x", pady=4)
        self.record_btn = ttk.Button(self.train_box, text="Record point",
                                     command=self._record_point)
        self.record_btn.pack(fill="x")
        ttk.Button(self.train_box, text="Clear points",
                   command=self._clear_points).pack(fill="x", pady=(4, 0))
        row = ttk.Frame(self.train_box)
        row.pack(fill="x", pady=(4, 0))
        ttk.Button(row, text="Save…", command=self._save).pack(side="left", expand=True, fill="x")
        ttk.Button(row, text="Load…", command=self._load).pack(side="left", expand=True, fill="x")

        # ---- Live readouts ----
        read = ttk.LabelFrame(controls, text="Live", padding=8)
        read.pack(fill="x", pady=4)
        self.sensor_readout = ttk.Label(read, text="Sensor: —")
        self.sensor_readout.pack(anchor="w")
        self.pos_readout = ttk.Label(read, text="Motor: —")
        self.pos_readout.pack(anchor="w")
        self.target_readout = ttk.Label(read, text="Target: —")
        self.target_readout.pack(anchor="w")
        self.count_readout = ttk.Label(read, text="Recorded points: 0")
        self.count_readout.pack(anchor="w")

        # ---- Right column: the graph ----
        graph = ttk.Frame(main)
        graph.pack(side="left", fill="both", expand=True)
        if HAS_MPL:
            self.fig = Figure(figsize=(5.5, 4.5), dpi=100)
            self.ax = self.fig.add_subplot(111)
            self.canvas = FigureCanvasTkAgg(self.fig, master=graph)
            self.canvas.get_tk_widget().pack(fill="both", expand=True)
        else:
            self.fig = None
            ttk.Label(
                graph,
                text=("matplotlib is not installed, so the graph is unavailable.\n"
                      "Install it with:  pip install matplotlib\n\n"
                      "Training and running still work; watch the Live panel."),
                justify="left", padding=20,
            ).pack(fill="both", expand=True)

    # ------------------------------------------------------------- helpers ---
    def _motor_labels(self):
        return self.backend.motor_labels

    def _backend_key(self):
        if isinstance(self.backend, SimulatedBackend):
            return "sim"
        return self.backend.sensor_kind  # "color" | "controller"

    def _selected_features(self):
        """Currently ticked features, in backend order. Falls back to the first
        feature if nothing is ticked. GUI thread only (reads Tk variables)."""
        feats = [f for f in self.backend.sensor_features
                 if f in self.feature_vars and self.feature_vars[f].get()]
        if not feats:
            feats = [self.backend.sensor_features[0]]
        return feats

    def _commit_features(self, feats):
        """Record the accepted selection for both the GUI and the worker thread."""
        self._committed_features = list(feats)
        self._active_features = list(feats)   # atomic swap read by the worker

    def _rebuild_backend(self):
        """Construct the backend object matching the current UI selections."""
        if self.backend.connected:
            return
        if self.backend_var.get() == SimulatedBackend.name:
            self.backend = SimulatedBackend()
        else:
            motor_kind = "single" if self.motor_var.get() == "Single Motor" else "double"
            sensor_kind = "color" if self.sensor_kind_var.get() == "Color Sensor" else "controller"
            self.backend = LegoBackend(sensor_kind=sensor_kind, motor_kind=motor_kind)

        self.model.clear()
        self._build_feature_checks()
        self._build_sim_sensor_sliders()
        self._update_hw_widgets()
        self._rebuild_plot_series()
        self._redraw_points()

    def _build_feature_checks(self):
        """Create a checkbox per sensor feature for the current backend."""
        for child in self.feature_check_frame.winfo_children():
            child.destroy()
        self.feature_vars = {}
        defaults = DEFAULT_FEATURES.get(self._backend_key(), [])
        for feat in self.backend.sensor_features:
            var = tk.BooleanVar(value=(feat in defaults))
            self.feature_vars[feat] = var
            ttk.Checkbutton(self.feature_check_frame, text=feat, variable=var,
                            command=self._on_features_changed).pack(anchor="w")
        # Commit the default selection.
        self._commit_features(self._selected_features())
        self.model.set_features(self._committed_features)
        self._refresh_display_combo()

    def _refresh_display_combo(self):
        feats = self._committed_features
        self.display_combo["values"] = feats
        if self.display_feature_var.get() not in feats:
            self.display_feature_var.set(feats[0])

    def _on_features_changed(self):
        """Handle a checkbox toggle: keep the model's feature set consistent."""
        new_features = self._selected_features()
        if new_features == self._committed_features:
            return
        if not self.model.set_features(new_features):
            # Points exist with a different feature set — changing clears them.
            if messagebox.askyesno(
                    "Change sensor inputs",
                    "Changing which sensor readings drive the motor will clear the "
                    "recorded points (they were recorded with different inputs). "
                    "Continue?"):
                self.model.clear()
                self.model.set_features(new_features)
            else:
                self._restore_feature_checks()
                return
        self._commit_features(new_features)
        self._refresh_display_combo()
        self._redraw_points()

    def _restore_feature_checks(self):
        for feat, var in self.feature_vars.items():
            var.set(feat in self._committed_features)

    def _update_hw_widgets(self):
        """Enable/disable hardware widgets based on backend + filter + connection."""
        connected = self.backend.connected
        is_lego = isinstance(self.backend, LegoBackend)

        sel_state = "disabled" if connected else "readonly"
        self.backend_combo.config(state=sel_state)
        self.motor_combo.config(state=sel_state if is_lego else "disabled")
        self.sensor_kind_combo.config(state=sel_state if is_lego else "disabled")
        self.filter_check.config(state=("disabled" if connected or not is_lego else "normal"))

        use_cards = is_lego and self.filter_var.get()
        card_state = "normal" if (use_cards and not connected) else "disabled"
        combo_state = "readonly" if (use_cards and not connected) else "disabled"
        for w in (self.motor_serial_entry, self.sensor_serial_entry):
            w.config(state=card_state)
        for w in (self.motor_color_combo, self.sensor_color_combo):
            w.config(state=combo_state)

        # Simulation sliders: only for the Simulated backend.
        is_sim = isinstance(self.backend, SimulatedBackend)
        sim_state = "normal" if is_sim else "disabled"
        for child in self.sim_sensor_frame.winfo_children():
            try:
                child.configure(state=sim_state)
            except tk.TclError:
                pass
        motor_slider_state = "normal" if (is_sim and self.mode == MODE_TRAINING) else "disabled"
        self.sim_motor_scale.config(state=motor_slider_state)

    def _build_sim_sensor_sliders(self):
        """Create one slider per sensor feature for the simulated backend."""
        for child in self.sim_sensor_frame.winfo_children():
            child.destroy()
        self.sim_sensor_vars = {}
        if not isinstance(self.backend, SimulatedBackend):
            return
        for feat in self.backend.sensor_features:
            ttk.Label(self.sim_sensor_frame, text=feat).pack(anchor="w")
            var = tk.DoubleVar(value=50)
            self.sim_sensor_vars[feat] = var
            ttk.Scale(self.sim_sensor_frame, from_=0, to=100, variable=var,
                      command=lambda v, f=feat: self._on_sim_sensor(f)).pack(fill="x")
            self.backend.set_sim_sensor(feat, 50)

    # ---------------------------------------------------------------- plot ---
    def _rebuild_plot_series(self):
        if not self.fig:
            return
        self.ax.clear()
        self.ax.set_title("Sensor reading vs. Motor position")
        self.ax.set_ylabel("Motor position (deg)")
        self.ax.grid(True, alpha=0.3)
        self.series = []
        for i, lab in enumerate(self._motor_labels()):
            c = SERIES_COLORS[i % len(SERIES_COLORS)]
            (scatter,) = self.ax.plot([], [], "o", color=c, label=f"{lab} points")
            (marker,) = self.ax.plot([], [], "*", color=c, markersize=15)
            (pred,) = self.ax.plot([], [], "--", color=c, alpha=0.6)
            self.series.append({"scatter": scatter, "marker": marker, "pred": pred})
        self.ax.legend(loc="upper left", fontsize=8)
        self._rescale_axes()

    def _display_index(self, features):
        """Index of the graph x-axis feature within `features` (0 if absent)."""
        disp = self.display_feature_var.get()
        return features.index(disp) if disp in features else 0

    def _rescale_axes(self):
        if not self.fig:
            return
        pts = self.model.points()
        di = self._display_index(self.model.features)
        xs = [p[0][di] for p in pts if len(p[0]) > di]
        ys = [v for p in pts for v in p[1]]
        xmin, xmax = (min(xs), max(xs)) if xs else (0, 100)
        ymin, ymax = (min(ys), max(ys)) if ys else (-180, 180)
        xpad = max((xmax - xmin) * 0.1, 5)
        ypad = max((ymax - ymin) * 0.1, 10)
        self.ax.set_xlim(xmin - xpad, xmax + xpad)
        self.ax.set_ylim(ymin - ypad, ymax + ypad)
        self.ax.set_xlabel(f"Sensor reading ({self.display_feature_var.get()})")
        self.canvas.draw_idle()

    def _redraw_points(self):
        if not self.fig:
            return
        pts = self.model.points()
        di = self._display_index(self.model.features)
        for i, s in enumerate(self.series):
            xs = [p[0][di] for p in pts if len(p[0]) > di and len(p[1]) > i]
            ys = [p[1][i] for p in pts if len(p[0]) > di and len(p[1]) > i]
            s["scatter"].set_data(xs, ys)
        self._rescale_axes()

    # ------------------------------------------------------------ callbacks --
    def _on_sim_sensor(self, feature):
        if isinstance(self.backend, SimulatedBackend):
            self.backend.set_sim_sensor(feature, self.sim_sensor_vars[feature].get())

    def _on_sim_motor(self, _value=None):
        if isinstance(self.backend, SimulatedBackend) and self.mode == MODE_TRAINING:
            self.backend.set_manual_motor(self.sim_motor_var.get())

    def _card_from(self, color_var, serial_var):
        serial = serial_var.get().strip()
        card = {"card_color": getattr(le, f"LEGO_COLOR_{color_var.get()}")}
        if serial:
            card["card_serial"] = serial
        return card

    def _on_connect(self):
        if self.backend.connected:
            self._stop_worker()
            self.backend.disconnect()
            self.conn_status.config(text="Not connected", foreground="#a00")
            self.connect_btn.config(text="Connect")
            self._update_hw_widgets()
            return

        try:
            if isinstance(self.backend, LegoBackend):
                if self.filter_var.get():
                    motor_card = self._card_from(self.motor_color_var, self.motor_serial_var)
                    sensor_card = self._card_from(self.sensor_color_var, self.sensor_serial_var)
                else:
                    motor_card = sensor_card = None
                self.backend.connect(motor_card=motor_card, sensor_card=sensor_card)
            else:
                self.backend.connect()
        except Exception as exc:  # noqa: BLE001 — surface any connection error
            messagebox.showerror("Connection failed", str(exc))
            return

        self.conn_status.config(text="Connected", foreground="#0a0")
        self.connect_btn.config(text="Disconnect")
        self._update_hw_widgets()
        self._start_worker()

    def _toggle_mode(self):
        if self.mode == MODE_TRAINING:
            if len(self.model) == 0:
                if not messagebox.askyesno(
                        "No training data",
                        "You haven't recorded any points. Run mode won't move the "
                        "motor until points exist. Switch anyway?"):
                    return
            self.mode = MODE_RUN
        else:
            self.mode = MODE_TRAINING
        self._update_mode_ui()

    def _update_mode_ui(self):
        self.mode_label.config(text=f"Current: {self.mode}")
        if self.mode == MODE_TRAINING:
            self.mode_btn.config(text="Switch to RUN mode")
            self.record_btn.config(state="normal")
            self.sim_motor_label.config(text="Motor position (training)")
        else:
            self.mode_btn.config(text="Switch to TRAINING mode")
            self.record_btn.config(state="disabled")
            self.sim_motor_label.config(text="Motor position (auto in run)")
        self._update_hw_widgets()

    def _record_point(self):
        if not self.backend.connected:
            messagebox.showinfo("Not connected", "Connect a backend first.")
            return
        with self._state_lock:
            vec = list(self._latest_vec)
            feats = list(self._latest_features)
            pos = list(self._latest_pos)
        # Make sure the model records against the current feature set.
        if feats and not self.model.set_features(feats):
            # Should not happen (features stay in sync), but guard anyway.
            self.model.clear()
            self.model.set_features(feats)
        self.model.add_point(vec, pos)
        self._redraw_points()

    def _clear_points(self):
        if messagebox.askyesno("Clear", "Remove all recorded points?"):
            self.model.clear()
            self._redraw_points()

    def _save(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".json", filetypes=[("JSON", "*.json")],
            title="Save training points")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.model.to_json())

    def _load(self):
        path = filedialog.askopenfilename(
            filetypes=[("JSON", "*.json")], title="Load training points")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                self.model.load_json(f.read())
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Load failed", str(exc))
            return
        # Reflect the loaded feature set in the checkboxes where possible.
        loaded = self.model.features
        for feat, var in self.feature_vars.items():
            var.set(feat in loaded)
        self._commit_features(self._selected_features())
        self._refresh_display_combo()
        self._redraw_points()

    # -------------------------------------------------- background control ---
    def _start_worker(self):
        self._worker_stop.clear()
        self._worker = threading.Thread(target=self._control_loop, daemon=True)
        self._worker.start()

    def _stop_worker(self):
        self._worker_stop.set()
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=1.0)
        self._worker = None

    def _control_loop(self):
        """Runs off the GUI thread. Reads the sensor/motor and, in run mode,
        drives the motor toward the nearest-neighbor prediction."""
        while not self._worker_stop.is_set():
            try:
                # Read the plain-list mirror, never the Tk variables, off-thread.
                features = self._active_features or self.backend.sensor_features[:1]
                vec = self.backend.read_sensor_vector(features)
                pos = self.backend.read_motor_position()
                target = None
                if self.mode == MODE_RUN:
                    target = self.model.predict(vec)
                    if target is not None:
                        self.backend.move_motor_to(target)
                with self._state_lock:
                    self._latest_vec = vec
                    self._latest_features = features
                    self._latest_pos = pos
                    self._latest_target = target
            except Exception:  # noqa: BLE001 — keep the loop alive on transient errors
                pass
            self._worker_stop.wait(0.1)

    # -------------------------------------------------------------- GUI tick -
    def _tick(self):
        with self._state_lock:
            vec = list(self._latest_vec)
            feats = list(self._latest_features)
            pos = list(self._latest_pos)
            target = list(self._latest_target) if self._latest_target is not None else None

        connected = self.backend.connected
        if connected and feats:
            self.sensor_readout.config(
                text="Sensor: " + ", ".join(f"{f}={v:.0f}" for f, v in zip(feats, vec)))
        else:
            self.sensor_readout.config(text="Sensor: —")
        self.pos_readout.config(
            text=("Motor: " + ", ".join(f"{v:.0f}°" for v in pos)) if connected else "Motor: —")
        self.target_readout.config(
            text=("Target: " + ", ".join(f"{v:.0f}°" for v in target))
            if (connected and target is not None) else "Target: —")
        self.count_readout.config(text=f"Recorded points: {len(self.model)}")

        if self.fig and connected and feats:
            di = feats.index(self.display_feature_var.get()) \
                if self.display_feature_var.get() in feats else 0
            x = vec[di] if di < len(vec) else 0
            for i, s in enumerate(self.series):
                if i < len(pos):
                    s["marker"].set_data([x], [pos[i]])
                    if self.mode == MODE_RUN and target is not None and i < len(target):
                        s["pred"].set_data([x, x], [pos[i], target[i]])
                    else:
                        s["pred"].set_data([], [])
            self.canvas.draw_idle()

        self.root.after(self.POLL_MS, self._tick)

    def _on_close(self):
        self._stop_worker()
        try:
            if self.backend.connected:
                self.backend.disconnect()
        finally:
            self.root.destroy()


def main():
    if not HAS_MPL:
        print("Note: matplotlib is not installed; the graph will be disabled.")
        print("Install it with:  pip install matplotlib")
    root = tk.Tk()
    SmartMotorApp(root)
    root.minsize(860, 600)
    root.mainloop()


if __name__ == "__main__":
    main()
