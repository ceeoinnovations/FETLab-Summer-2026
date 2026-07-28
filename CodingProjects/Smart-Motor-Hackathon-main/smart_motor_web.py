"""
Smart Motor (Web) — a teachable sensor-to-motor interface in your browser.

This is the browser version of `smart_motor.py`. It runs a small local web
server (Python standard library only) that owns the Bluetooth connection to the
LEGO(R) Education hardware, and serves a web page you control from your browser.

Why a local server (and not a plain website)?
----------------------------------------------
The LEGO Education hardware talks over Bluetooth Low Energy using the
`legoeducation` Python package. A web page on its own cannot speak that
protocol, so the browser UI here talks to this local Python program, which does
the Bluetooth work and relays sensor readings / motor commands. Everything stays
on your machine (default http://127.0.0.1:8000).

How it works (same model as the desktop app)
---------------------------------------------
* TRAINING MODE — record points pairing the current sensor reading(s) with the
  motor position(s). Points are drawn on a live graph.
* RUN MODE — the live readings are matched to the closest recorded point
  (nearest-neighbor) and the motor is driven there.

Mapping (Double Motor)
----------------------
* COMBINED (default) — all selected sensor features drive all motors together
  (one nearest-neighbor model over the full reading vector).
* INDEPENDENT — each motor axis is driven by ONE assignable input feature, with
  its own separate model. For a Controller this defaults to left motor <-
  `leftPercent` and right motor <- `rightPercent`, but you can assign any lever
  to any motor (e.g. swap them). Each side is trained and predicted separately,
  so in run mode both levers act at once, like a normal controller: press the
  left lever and its motor turns, press the right lever and its motor turns,
  press both and both turn.

Requirements
------------
    pip install legoeducation      # only needed for the LEGO Hardware backend

(The web UI and graph need no extra packages — just a browser.)

Run it
------
    python smart_motor_web.py

then open the printed URL (http://127.0.0.1:8000) in Chrome/Edge/Firefox.
Use  --port N  to change the port, and  --no-browser  to not auto-open.
"""

import argparse
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

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
    """Stores (sensor_vector, [positions...]) points; predicts by closest match."""

    def __init__(self):
        self._points = []
        self._features = []
        self._lock = threading.Lock()

    @property
    def features(self):
        with self._lock:
            return list(self._features)

    def set_features(self, features):
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
        with self._lock:
            if not self._points:
                return None

            def dist2(point):
                return sum((a - b) ** 2 for a, b in zip(point[0], sensor_vector))

            return list(min(self._points, key=dist2)[1])

    def to_json_obj(self):
        return {"features": self.features, "points": self.points()}

    def load_obj(self, data):
        pts = data.get("points", [])
        features = data.get("features", [])
        parsed = []
        for entry in pts:
            s, p = entry
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
# Backends
# =============================================================================
class Backend:
    name = "base"
    sensor_features = ["value"]
    motor_labels = ["motor"]

    def connect(self, **kwargs):
        raise NotImplementedError

    def disconnect(self):
        pass

    @property
    def connected(self):
        return False

    def read_sensor_vector(self, features):
        raise NotImplementedError

    def read_motor_position(self):
        raise NotImplementedError

    def move_motor_to(self, positions):
        raise NotImplementedError


class SimulatedBackend(Backend):
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

    def set_sim_sensor(self, feature, value):
        with self._lock:
            self._sensor[feature] = float(value)

    def set_manual_motor(self, value):
        with self._lock:
            self._motor_pos = float(value)
            self._motor_target = float(value)

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
    name = "LEGO Hardware"
    COLOR_FEATURES = ["reflection", "hue", "color", "value", "saturation"]
    CONTROLLER_FEATURES = ["leftPercent", "rightPercent", "leftAngle", "rightAngle"]

    def __init__(self, sensor_kind="color", motor_kind="single"):
        self.sensor_kind = sensor_kind
        self.motor_kind = motor_kind
        self.sensor_features = (
            self.COLOR_FEATURES if sensor_kind == "color" else self.CONTROLLER_FEATURES
        )
        self.motor_labels = ["motor"] if motor_kind == "single" else ["left", "right"]
        self.motor = None
        self.sensor = None

    def connect(self, motor_card=None, sensor_card=None):
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
# Shared application state (headless — driven by the web API)
# =============================================================================
MODE_TRAINING = "TRAINING"
MODE_RUN = "RUN"
MAP_COMBINED = "combined"
MAP_INDEPENDENT = "independent"

DEFAULT_FEATURES = {
    "sim": ["valueX"],
    "color": ["reflection"],
    "controller": ["leftPercent", "rightPercent"],
}
COLOR_NAMES = ["GREEN", "BLUE", "RED", "ORANGE", "YELLOW", "AZURE", "PURPLE", "MAGENTA"]


class SmartMotorState:
    def __init__(self):
        self.backend = SimulatedBackend()
        self.mode = MODE_TRAINING
        self.mapping = MAP_COMBINED

        # Combined-mode model + its selection / graph axis.
        self.model = NearestNeighborModel()
        self.combined_features = []
        self.display_feature = None

        # Independent-mode: one model + one input feature per motor axis.
        self.axis_features = []
        self.axis_models = []

        # Resolved list of features the worker actually reads.
        self.active_features = []

        self._latest_vec = [0.0]
        self._latest_features = []
        self._latest_pos = [0.0]
        self._latest_target = None
        self._state_lock = threading.Lock()

        self._worker = None
        self._worker_stop = threading.Event()

        self._apply_defaults()

    # -- helpers --
    def _backend_key(self):
        if isinstance(self.backend, SimulatedBackend):
            return "sim"
        return self.backend.sensor_kind

    def _num_axes(self):
        return len(self.backend.motor_labels)

    def _supports_independent(self):
        return self._num_axes() >= 2

    def _default_axis_features(self):
        feats = self.backend.sensor_features
        return [feats[min(i, len(feats) - 1)] for i in range(self._num_axes())]

    def _apply_defaults(self):
        # Combined selection.
        key = self._backend_key()
        cf = [f for f in self.backend.sensor_features if f in DEFAULT_FEATURES.get(key, [])]
        if not cf:
            cf = [self.backend.sensor_features[0]]
        self.combined_features = cf
        self.display_feature = cf[0]
        self.model = NearestNeighborModel()
        self.model.set_features(cf)

        # Per-axis models.
        self.axis_features = self._default_axis_features()
        self.axis_models = [NearestNeighborModel() for _ in range(self._num_axes())]
        for a, feat in enumerate(self.axis_features):
            self.axis_models[a].set_features([feat])

        if not self._supports_independent():
            self.mapping = MAP_COMBINED
        self._sync_active_features()

    def _sync_active_features(self):
        if self.mapping == MAP_INDEPENDENT and self._supports_independent():
            seen = []
            for f in self.axis_features:
                if f not in seen:
                    seen.append(f)
            self.active_features = seen
        else:
            self.active_features = list(self.combined_features)

    # -- configuration --
    def set_config(self, backend_name, motor, sensor):
        if self.backend.connected:
            raise RuntimeError("Disconnect before changing the configuration.")
        if backend_name == SimulatedBackend.name:
            self.backend = SimulatedBackend()
        else:
            motor_kind = "single" if motor == "Single Motor" else "double"
            sensor_kind = "color" if sensor == "Color Sensor" else "controller"
            self.backend = LegoBackend(sensor_kind=sensor_kind, motor_kind=motor_kind)
        self._apply_defaults()

    def set_mapping(self, mode):
        self.mapping = MAP_INDEPENDENT if (mode == MAP_INDEPENDENT and self._supports_independent()) else MAP_COMBINED
        self._sync_active_features()

    def set_features(self, features):
        features = [f for f in features if f in self.backend.sensor_features]
        if not features:
            features = [self.backend.sensor_features[0]]
        if not self.model.set_features(features):
            self.model.clear()
            self.model.set_features(features)
        self.combined_features = features
        if self.display_feature not in features:
            self.display_feature = features[0]
        self._sync_active_features()

    def set_axis_feature(self, axis, feature):
        if not (0 <= axis < self._num_axes()):
            return
        if feature not in self.backend.sensor_features:
            return
        if self.axis_features[axis] != feature:
            # A different input drives this motor now — its old points no longer apply.
            self.axis_models[axis].clear()
            self.axis_models[axis].set_features([feature])
            self.axis_features[axis] = feature
        self._sync_active_features()

    def set_display_feature(self, feature):
        if feature in self.combined_features:
            self.display_feature = feature

    def set_mode(self, mode):
        self.mode = MODE_RUN if mode == MODE_RUN else MODE_TRAINING

    # -- connection --
    def connect(self, cfg):
        if isinstance(self.backend, LegoBackend):
            if cfg.get("filter"):
                motor_card = self._card(cfg.get("motor_color"), cfg.get("motor_serial"))
                sensor_card = self._card(cfg.get("sensor_color"), cfg.get("sensor_serial"))
            else:
                motor_card = sensor_card = None
            self.backend.connect(motor_card=motor_card, sensor_card=sensor_card)
        else:
            self.backend.connect()
        self._start_worker()

    def _card(self, color_name, serial):
        card = {"card_color": getattr(le, f"LEGO_COLOR_{color_name}")}
        serial = (serial or "").strip()
        if serial:
            card["card_serial"] = serial
        return card

    def disconnect(self):
        self._stop_worker()
        if self.backend.connected:
            self.backend.disconnect()

    # -- training / run actions --
    def record(self, axis=None):
        if not self.backend.connected:
            raise RuntimeError("Not connected.")
        with self._state_lock:
            vec = list(self._latest_vec)
            feats = list(self._latest_features)
            pos = list(self._latest_pos)

        if self.mapping == MAP_INDEPENDENT and self._supports_independent():
            axes = [axis] if axis is not None else range(self._num_axes())
            for a in axes:
                if not (0 <= a < self._num_axes()) or a >= len(pos):
                    continue
                feat = self.axis_features[a]
                if feat not in feats:
                    continue
                val = vec[feats.index(feat)]
                if not self.axis_models[a].set_features([feat]):
                    self.axis_models[a].clear()
                    self.axis_models[a].set_features([feat])
                # Only this motor's own position is recorded — not the other side.
                self.axis_models[a].add_point([val], [pos[a]])
        else:
            if feats and not self.model.set_features(feats):
                self.model.clear()
                self.model.set_features(feats)
            self.model.add_point(vec, pos)

    def clear_points(self, axis=None):
        if self.mapping == MAP_INDEPENDENT and self._supports_independent():
            if axis is None:
                for m in self.axis_models:
                    m.clear()
            elif 0 <= axis < self._num_axes():
                self.axis_models[axis].clear()
        else:
            self.model.clear()

    def sim_sensor(self, feature, value):
        if isinstance(self.backend, SimulatedBackend):
            self.backend.set_sim_sensor(feature, value)

    def sim_motor(self, value):
        if isinstance(self.backend, SimulatedBackend) and self.mode == MODE_TRAINING:
            self.backend.set_manual_motor(value)

    # -- persistence --
    def export_json(self):
        return json.dumps({
            "mapping": self.mapping,
            "combined": self.model.to_json_obj(),
            "display": self.display_feature,
            "axes": [{"feature": self.axis_features[a],
                      "model": self.axis_models[a].to_json_obj()}
                     for a in range(self._num_axes())],
        }, indent=2)

    def import_json(self, text):
        data = json.loads(text)
        # Back-compat: a plain {features, points} file loads into the combined model.
        if "combined" not in data and "points" in data:
            self.mapping = MAP_COMBINED
            self.model.load_obj(data)
            loaded = self.model.features
            feats = [f for f in self.backend.sensor_features if f in loaded]
            if feats:
                self.combined_features = feats
                self.display_feature = feats[0]
            self._sync_active_features()
            return

        self.mapping = data.get("mapping", MAP_COMBINED)
        if self.mapping == MAP_INDEPENDENT and not self._supports_independent():
            self.mapping = MAP_COMBINED

        self.model.load_obj(data.get("combined", {"features": [], "points": []}))
        cfeats = [f for f in self.backend.sensor_features if f in self.model.features]
        if cfeats:
            self.combined_features = cfeats
        disp = data.get("display")
        self.display_feature = disp if disp in self.combined_features else self.combined_features[0]

        axes = data.get("axes", [])
        for a in range(self._num_axes()):
            if a < len(axes):
                feat = axes[a].get("feature")
                if feat in self.backend.sensor_features:
                    self.axis_features[a] = feat
                self.axis_models[a].load_obj(axes[a].get("model", {"features": [], "points": []}))
        self._sync_active_features()

    # -- background control loop --
    def _start_worker(self):
        self._worker_stop.clear()
        self._worker = threading.Thread(target=self._control_loop, daemon=True)
        self._worker.start()

    def _stop_worker(self):
        self._worker_stop.set()
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=1.0)
        self._worker = None

    def _independent_targets(self, features, vec, pos):
        """Per-axis nearest-neighbor: each motor follows its own input feature.
        Returns (target_list, move_list, any_prediction). Untrained axes hold
        their current position so the other side can still move."""
        target, move, any_pred = [], [], False
        for a in range(self._num_axes()):
            feat = self.axis_features[a]
            pred = None
            if feat in features:
                val = vec[features.index(feat)]
                r = self.axis_models[a].predict([val])
                if r is not None:
                    pred = r[0]
            target.append(pred)
            move.append(pred if pred is not None else (pos[a] if a < len(pos) else 0.0))
            if pred is not None:
                any_pred = True
        return target, move, any_pred

    def _control_loop(self):
        while not self._worker_stop.is_set():
            try:
                features = self.active_features or self.backend.sensor_features[:1]
                vec = self.backend.read_sensor_vector(features)
                pos = self.backend.read_motor_position()
                target = None
                if self.mode == MODE_RUN:
                    if self.mapping == MAP_INDEPENDENT and self._supports_independent():
                        target, move, any_pred = self._independent_targets(features, vec, pos)
                        if any_pred:
                            self.backend.move_motor_to(move)
                    else:
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

    # -- snapshot for the browser --
    def _build_plot(self, vec, feats, pos, target, connected):
        labels = self.backend.motor_labels
        series = []
        if self.mapping == MAP_INDEPENDENT and self._supports_independent():
            xlabel = "input value"
            for a, lab in enumerate(labels):
                feat = self.axis_features[a]
                pts = self.axis_models[a].points()
                spoints = [[p[0][0], p[1][0]] for p in pts if p[0] and p[1]]
                live, ty = None, None
                if connected and feat in feats:
                    lx = vec[feats.index(feat)]
                    if a < len(pos):
                        live = [lx, pos[a]]
                    if target is not None and a < len(target) and target[a] is not None:
                        ty = target[a]
                series.append({"label": f"{lab} ← {feat}", "points": spoints,
                               "live": live, "targetY": ty})
        else:
            disp = self.display_feature
            mfeats = self.model.features
            di = mfeats.index(disp) if disp in mfeats else 0
            pts = self.model.points()
            xlabel = disp or ""
            for a, lab in enumerate(labels):
                spoints = [[p[0][di], p[1][a]] for p in pts
                           if len(p[0]) > di and len(p[1]) > a]
                live, ty = None, None
                if connected and disp in feats:
                    lx = vec[feats.index(disp)]
                    if a < len(pos):
                        live = [lx, pos[a]]
                    if target is not None and a < len(target) and target[a] is not None:
                        ty = target[a]
                series.append({"label": lab, "points": spoints, "live": live, "targetY": ty})
        return {"xlabel": xlabel, "series": series}

    def _points_label(self):
        if self.mapping == MAP_INDEPENDENT and self._supports_independent():
            parts = [f"{self.backend.motor_labels[a]}={len(self.axis_models[a])}"
                     for a in range(self._num_axes())]
            return "Recorded points — " + ", ".join(parts)
        return f"Recorded points: {len(self.model)}"

    def snapshot(self):
        with self._state_lock:
            vec = list(self._latest_vec)
            feats = list(self._latest_features)
            pos = list(self._latest_pos)
            target = list(self._latest_target) if self._latest_target is not None else None
        connected = self.backend.connected
        readings = {f: v for f, v in zip(feats, vec)} if connected else {}
        return {
            "hasLego": HAS_LE,
            "backend": self.backend.name,
            "isSim": isinstance(self.backend, SimulatedBackend),
            "motorKind": getattr(self.backend, "motor_kind", "single"),
            "sensorKind": getattr(self.backend, "sensor_kind", None),
            "connected": connected,
            "mode": self.mode,
            "mapping": self.mapping,
            "numAxes": self._num_axes(),
            "motorLabels": list(self.backend.motor_labels),
            "availableFeatures": list(self.backend.sensor_features),
            "selectedFeatures": list(self.combined_features),
            "displayFeature": self.display_feature,
            "axisFeatures": list(self.axis_features),
            "colorNames": COLOR_NAMES,
            "readings": readings,
            "positions": pos if connected else [],
            "target": target if connected else None,
            "plot": self._build_plot(vec, feats, pos, target, connected),
            "pointsLabel": self._points_label(),
        }


# =============================================================================
# The web page (single self-contained HTML document; canvas graph, no CDNs)
# =============================================================================
PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Smart Motor — teachable sensor-to-motor</title>
<style>
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }
  body { margin: 0; font: 14px/1.4 system-ui, sans-serif; background: #f4f5f7; color: #1a1a1a; }
  @media (prefers-color-scheme: dark) {
    body { background: #1e1f22; color: #e6e6e6; }
    .card { background: #2a2b2f !important; border-color: #3a3b40 !important; }
    input, select, button { background: #34353a; color: #e6e6e6; border-color: #4a4b50; }
    .muted { color: #9aa0a6 !important; }
  }
  header { padding: 12px 16px; background: #d1170f; color: #fff; display: flex; align-items: center; gap: 12px; }
  header h1 { font-size: 18px; margin: 0; }
  header .status { margin-left: auto; font-weight: 600; }
  .wrap { display: flex; gap: 16px; padding: 16px; align-items: flex-start; flex-wrap: wrap; }
  .col-controls { flex: 0 0 330px; display: flex; flex-direction: column; gap: 12px; }
  .col-graph { flex: 1 1 480px; min-width: 360px; }
  .card { background: #fff; border: 1px solid #e0e0e0; border-radius: 8px; padding: 12px; }
  .card h2 { font-size: 13px; text-transform: uppercase; letter-spacing: .04em; margin: 0 0 8px; color: #666; }
  label.row { display: grid; grid-template-columns: 96px 1fr; align-items: center; gap: 8px; margin: 4px 0; }
  input, select, button { padding: 6px 8px; border: 1px solid #cfcfcf; border-radius: 6px; font: inherit; }
  button { cursor: pointer; background: #f0f0f0; }
  button.primary { background: #d1170f; color: #fff; border-color: #d1170f; }
  /* Keep these secondary buttons readable in both light and dark themes. */
  #modeBtn, #clearBtn, #saveBtn, #loadBtn, .lightbtn {
    background: #ffffff; color: #000000; border: 1px solid #b8b8b8;
  }
  #modeBtn:hover, #clearBtn:hover, #saveBtn:hover, #loadBtn:hover, .lightbtn:hover { background: #ececec; }
  button.wide { width: 100%; }
  button:disabled { opacity: .5; cursor: not-allowed; }
  .btn-row { display: flex; gap: 8px; }
  .btn-row > * { flex: 1; }
  .checks label { display: block; margin: 3px 0; cursor: pointer; }
  .slider { display: block; width: 100%; }
  .readout { font-variant-numeric: tabular-nums; margin: 2px 0; }
  .muted { color: #888; }
  canvas { width: 100%; height: auto; background: #fff; border: 1px solid #e0e0e0; border-radius: 8px; }
  @media (prefers-color-scheme: dark) { canvas { background: #26272b; } }
  .pill { display: inline-block; padding: 1px 8px; border-radius: 999px; font-size: 12px; font-weight: 600; }
  .pill.train { background: #ffe9b3; color: #7a5b00; }
  .pill.run { background: #bce8c6; color: #0b5d24; }
  .hint { font-size: 12px; color: #888; margin-top: 6px; }
  .axisrow { display: grid; grid-template-columns: 1fr; gap: 4px; margin: 6px 0; padding: 6px; border: 1px dashed #ccc; border-radius: 6px; }
  .hidden { display: none; }
</style>
</head>
<body>
<header>
  <h1>Smart Motor</h1>
  <span class="muted">teachable sensor-to-motor</span>
  <span class="status" id="connStatus">Not connected</span>
</header>

<div class="wrap">
  <div class="col-controls">
    <div class="card">
      <h2>Hardware</h2>
      <label class="row">Backend
        <select id="backendSel"></select>
      </label>
      <label class="row" id="motorRow">Motor
        <select id="motorSel">
          <option>Single Motor</option><option>Double Motor</option>
        </select>
      </label>
      <label class="row" id="sensorRow">Sensor
        <select id="sensorSel">
          <option>Color Sensor</option><option>Controller</option>
        </select>
      </label>
      <div id="filterWrap">
        <label style="display:block;margin:6px 0;">
          <input type="checkbox" id="filterChk"> Filter by connection card
        </label>
        <label class="row">Motor card
          <span><select id="motorColor" class="cardsel"></select>
                <input id="motorSerial" placeholder="serial" style="width:70px"></span>
        </label>
        <label class="row">Sensor card
          <span><select id="sensorColor" class="cardsel"></select>
                <input id="sensorSerial" placeholder="serial" style="width:70px"></span>
        </label>
      </div>
      <button class="primary wide" id="connectBtn" style="margin-top:8px">Connect</button>
      <div class="hint" id="connHint">Leave the filter off to connect to the first motor and sensor found.</div>
    </div>

    <div class="card">
      <h2>Sensor inputs</h2>
      <label class="row" id="mappingRow" style="display:none">Mapping
        <select id="mappingSel">
          <option value="combined">Combined (all inputs → all motors)</option>
          <option value="independent">Independent (one input per motor)</option>
        </select>
      </label>

      <div id="combinedInputs">
        <div class="muted" style="font-size:12px">Tick every reading that should drive the motor:</div>
        <div class="checks" id="featureChecks"></div>
        <label class="row" style="margin-top:6px">Graph x-axis
          <select id="displaySel"></select>
        </label>
      </div>

      <div id="independentInputs" style="display:none">
        <div class="muted" style="font-size:12px">Assign one input to each motor (any lever → any motor):</div>
        <div id="axisFeatureRows"></div>
      </div>
    </div>

    <div class="card" id="simCard">
      <h2>Simulation inputs</h2>
      <div id="simSensors"></div>
      <div id="simMotorWrap">
        <div class="muted" id="simMotorLabel" style="font-size:12px">Motor position (training)</div>
        <input type="range" min="-180" max="180" value="0" id="simMotor" class="slider">
      </div>
    </div>

    <div class="card">
      <h2>Mode <span id="modePill" class="pill train">TRAINING</span></h2>
      <button class="wide" id="modeBtn">Switch to RUN mode</button>
    </div>

    <div class="card">
      <h2>Training</h2>
      <div id="combinedTraining">
        <button class="wide primary" id="recordBtn">Record point</button>
      </div>
      <div id="independentTraining" style="display:none">
        <div class="hint">Record Left and Record Right each store a point for that motor only.</div>
        <div id="axisRecordRows"></div>
      </div>
      <button class="wide" id="clearBtn" style="margin-top:6px">Clear all points</button>
      <div class="btn-row" style="margin-top:6px">
        <button id="saveBtn">Save…</button>
        <button id="loadBtn">Load…</button>
        <input type="file" id="loadFile" accept="application/json" class="hidden">
      </div>
    </div>

    <div class="card">
      <h2>Live</h2>
      <div class="readout" id="rSensor">Sensor: —</div>
      <div class="readout" id="rMotor">Motor: —</div>
      <div class="readout" id="rTarget">Target: —</div>
      <div class="readout" id="rCount">Recorded points: 0</div>
    </div>
  </div>

  <div class="col-graph">
    <div class="card">
      <h2>Sensor reading vs. Motor position</h2>
      <canvas id="graph" width="640" height="480"></canvas>
      <div class="hint" id="legend"></div>
    </div>
  </div>
</div>

<script>
const SERIES_COLORS = ["#1f77b4", "#ff7f0e"];
let S = null;
let inited = false;

async function api(path, body) {
  const opt = { method: body ? "POST" : "GET" };
  if (body) { opt.headers = {"Content-Type": "application/json"}; opt.body = JSON.stringify(body); }
  const r = await fetch(path, opt);
  const t = await r.text();
  let data = {};
  try { data = t ? JSON.parse(t) : {}; }
  catch (e) { throw new Error("HTTP " + r.status + ": " + t); }
  if (!r.ok || data.error) throw new Error(data.error || ("HTTP " + r.status));
  return data;
}

function el(id) { return document.getElementById(id); }

async function pushConfig() {
  await api("/api/config", {
    backend: el("backendSel").value,
    motor: el("motorSel").value,
    sensor: el("sensorSel").value,
  }).catch(e => alert(e.message));
}

async function pushFeatures() {
  const feats = [...document.querySelectorAll(".featchk:checked")].map(c => c.value);
  await api("/api/features", { features: feats }).catch(e => alert(e.message));
}

function buildOnce(s) {
  const b = el("backendSel");
  b.innerHTML = "";
  const options = ["Simulated"].concat(s.hasLego ? ["LEGO Hardware"] : []);
  for (const o of options) { const op = document.createElement("option"); op.textContent = o; b.appendChild(op); }
  b.value = s.backend;
  for (const sel of [el("motorColor"), el("sensorColor")]) {
    sel.innerHTML = "";
    for (const c of s.colorNames) { const op = document.createElement("option"); op.textContent = c; sel.appendChild(op); }
    sel.value = "AZURE";
  }
  b.onchange = pushConfig;
  el("motorSel").onchange = pushConfig;
  el("sensorSel").onchange = pushConfig;
  el("filterChk").onchange = () => render();
  el("displaySel").onchange = () => api("/api/display", { feature: el("displaySel").value });
  el("mappingSel").onchange = () => api("/api/mapping", { mode: el("mappingSel").value });
  el("modeBtn").onclick = () => api("/api/mode", { mode: S.mode === "RUN" ? "TRAINING" : "RUN" });
  el("recordBtn").onclick = () => api("/api/record", {}).catch(e => alert(e.message));
  el("clearBtn").onclick = () => { if (confirm("Remove all recorded points?")) api("/api/clear", {}); };
  el("connectBtn").onclick = onConnect;
  el("simMotor").oninput = () => api("/api/sim", { motor: parseFloat(el("simMotor").value) });
  el("saveBtn").onclick = onSave;
  el("loadBtn").onclick = () => el("loadFile").click();
  el("loadFile").onchange = onLoad;
  inited = true;
}

async function onConnect() {
  if (S.connected) { await api("/api/disconnect", {}); return; }
  const body = {
    filter: el("filterChk").checked,
    motor_color: el("motorColor").value, motor_serial: el("motorSerial").value,
    sensor_color: el("sensorColor").value, sensor_serial: el("sensorSerial").value,
  };
  try { await api("/api/connect", body); }
  catch (e) { alert("Connection failed:\n" + e.message); }
}

async function onSave() {
  const r = await fetch("/api/export"); const text = await r.text();
  const blob = new Blob([text], {type: "application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = "smart_motor_points.json"; a.click();
  URL.revokeObjectURL(a.href);
}

function onLoad(ev) {
  const file = ev.target.files[0]; if (!file) return;
  const reader = new FileReader();
  reader.onload = () => api("/api/import", { text: reader.result }).catch(e => alert(e.message));
  reader.readAsText(file);
  ev.target.value = "";
}

let lastFeatureKey = "", lastSimKey = "", lastIndepKey = "";
function render() {
  const s = S; if (!s) return;
  if (!inited) buildOnce(s);

  el("connStatus").textContent = s.connected ? "Connected" : "Not connected";
  el("connStatus").style.color = s.connected ? "#bce8c6" : "#ffd0cd";
  el("connectBtn").textContent = s.connected ? "Disconnect" : "Connect";

  for (const id of ["backendSel","motorSel","sensorSel","filterChk","motorColor","motorSerial","sensorColor","sensorSerial"])
    el(id).disabled = s.connected;
  const showLego = (s.backend === "LEGO Hardware");
  el("motorRow").style.display = showLego ? "" : "none";
  el("sensorRow").style.display = showLego ? "" : "none";
  el("filterWrap").style.display = showLego ? "" : "none";
  const useCards = showLego && el("filterChk").checked && !s.connected;
  for (const id of ["motorColor","motorSerial","sensorColor","sensorSerial"]) el(id).disabled = !useCards;

  // mapping selector only when >1 motor axis
  const multi = s.numAxes > 1;
  el("mappingRow").style.display = multi ? "" : "none";
  el("mappingSel").value = s.mapping;
  const independent = multi && s.mapping === "independent";
  el("combinedInputs").style.display = independent ? "none" : "";
  el("independentInputs").style.display = independent ? "" : "none";
  el("combinedTraining").style.display = independent ? "none" : "";
  el("independentTraining").style.display = independent ? "" : "none";

  // combined feature checkboxes
  const featKey = s.availableFeatures.join(",") + "|" + s.selectedFeatures.join(",");
  if (featKey !== lastFeatureKey) {
    lastFeatureKey = featKey;
    const box = el("featureChecks"); box.innerHTML = "";
    for (const f of s.availableFeatures) {
      const lab = document.createElement("label");
      const cb = document.createElement("input");
      cb.type = "checkbox"; cb.className = "featchk"; cb.value = f;
      cb.checked = s.selectedFeatures.includes(f);
      cb.onchange = pushFeatures;
      lab.appendChild(cb); lab.appendChild(document.createTextNode(" " + f));
      box.appendChild(lab);
    }
    const dsel = el("displaySel"); dsel.innerHTML = "";
    for (const f of s.selectedFeatures) { const op = document.createElement("option"); op.textContent = f; dsel.appendChild(op); }
    dsel.value = s.displayFeature;
  }

  // independent per-axis controls (feature dropdown + record/clear per motor)
  const indepKey = s.numAxes + "|" + s.motorLabels.join(",") + "|" + s.availableFeatures.join(",");
  if (indepKey !== lastIndepKey) {
    lastIndepKey = indepKey;
    const frows = el("axisFeatureRows"); frows.innerHTML = "";
    const rrows = el("axisRecordRows"); rrows.innerHTML = "";
    for (let a = 0; a < s.numAxes; a++) {
      const lab = s.motorLabels[a];
      // feature assignment row
      const row = document.createElement("div"); row.className = "axisrow";
      const title = document.createElement("div"); title.innerHTML = "<b>" + lab + " motor</b> is driven by:";
      const sel = document.createElement("select"); sel.dataset.axis = a;
      for (const f of s.availableFeatures) { const op = document.createElement("option"); op.textContent = f; sel.appendChild(op); }
      sel.onchange = () => api("/api/axis_feature", { axis: a, feature: sel.value });
      row.appendChild(title); row.appendChild(sel); frows.appendChild(row);
      // record/clear row
      const rr = document.createElement("div"); rr.className = "btn-row"; rr.style.marginTop = "4px";
      const rec = document.createElement("button"); rec.className = "primary"; rec.textContent = "Record " + lab;
      rec.dataset.axis = a;
      rec.onclick = () => api("/api/record", { axis: a }).catch(e => alert(e.message));
      const clr = document.createElement("button"); clr.className = "lightbtn"; clr.textContent = "Clear " + lab;
      clr.onclick = () => api("/api/clear", { axis: a });
      rr.appendChild(rec); rr.appendChild(clr); rrows.appendChild(rr);
    }
  }
  // reflect current per-axis feature + disabled state each frame
  for (const sel of document.querySelectorAll("#axisFeatureRows select")) sel.value = s.axisFeatures[+sel.dataset.axis];
  for (const btn of document.querySelectorAll("#axisRecordRows .primary")) btn.disabled = (s.mode === "RUN") || !s.connected;

  // simulation inputs
  el("simCard").style.display = s.isSim ? "" : "none";
  if (s.isSim) {
    if (s.availableFeatures.join(",") !== lastSimKey) {
      lastSimKey = s.availableFeatures.join(",");
      const box = el("simSensors"); box.innerHTML = "";
      for (const f of s.availableFeatures) {
        const wrap = document.createElement("div");
        const lab = document.createElement("div"); lab.className = "muted"; lab.style.fontSize = "12px"; lab.textContent = f;
        const sl = document.createElement("input");
        sl.type = "range"; sl.min = 0; sl.max = 100; sl.value = 50; sl.className = "slider";
        sl.oninput = () => api("/api/sim", { feature: f, value: parseFloat(sl.value) });
        wrap.appendChild(lab); wrap.appendChild(sl); box.appendChild(wrap);
      }
    }
    el("simMotorLabel").textContent = s.mode === "RUN" ? "Motor position (auto in run)" : "Motor position (training)";
    el("simMotor").disabled = (s.mode === "RUN");
  }

  const pill = el("modePill");
  pill.textContent = s.mode; pill.className = "pill " + (s.mode === "RUN" ? "run" : "train");
  el("modeBtn").textContent = s.mode === "RUN" ? "Switch to TRAINING mode" : "Switch to RUN mode";
  el("recordBtn").disabled = (s.mode === "RUN") || !s.connected;

  if (s.connected && Object.keys(s.readings).length) {
    el("rSensor").textContent = "Sensor: " + Object.entries(s.readings).map(([k,v]) => k + "=" + v.toFixed(0)).join(", ");
  } else el("rSensor").textContent = "Sensor: —";
  el("rMotor").textContent = s.connected ? "Motor: " + s.positions.map(v => v.toFixed(0) + "°").join(", ") : "Motor: —";
  el("rTarget").textContent = (s.connected && s.target) ?
    "Target: " + s.target.map(v => v === null ? "—" : v.toFixed(0) + "°").join(", ") : "Target: —";
  el("rCount").textContent = s.pointsLabel;

  drawGraph(s);
}

function drawGraph(s) {
  const cv = el("graph"), ctx = cv.getContext("2d");
  const W = cv.width, H = cv.height;
  ctx.clearRect(0, 0, W, H);
  const padL = 54, padR = 16, padT = 14, padB = 40;
  const plot = s.plot, series = plot.series;

  const xs = [], ys = [];
  for (const ser of series) {
    for (const p of ser.points) { xs.push(p[0]); ys.push(p[1]); }
    if (ser.live) { xs.push(ser.live[0]); ys.push(ser.live[1]); }
    if (ser.targetY !== null && ser.targetY !== undefined) ys.push(ser.targetY);
  }
  let xmin = xs.length ? Math.min(...xs) : 0, xmax = xs.length ? Math.max(...xs) : 100;
  let ymin = ys.length ? Math.min(...ys) : -180, ymax = ys.length ? Math.max(...ys) : 180;
  const xpad = Math.max((xmax - xmin) * 0.1, 5), ypad = Math.max((ymax - ymin) * 0.1, 10);
  xmin -= xpad; xmax += xpad; ymin -= ypad; ymax += ypad;
  const X = v => padL + (v - xmin) / (xmax - xmin) * (W - padL - padR);
  const Y = v => H - padB - (v - ymin) / (ymax - ymin) * (H - padT - padB);

  ctx.strokeStyle = "rgba(128,128,128,0.25)"; ctx.fillStyle = getComputedStyle(document.body).color; ctx.lineWidth = 1;
  ctx.font = "11px system-ui, sans-serif";
  for (let i = 0; i <= 5; i++) {
    const gx = padL + i / 5 * (W - padL - padR);
    ctx.beginPath(); ctx.moveTo(gx, padT); ctx.lineTo(gx, H - padB); ctx.stroke();
    ctx.textAlign = "center"; ctx.fillText((xmin + i / 5 * (xmax - xmin)).toFixed(0), gx, H - padB + 14);
  }
  for (let i = 0; i <= 5; i++) {
    const gy = padT + i / 5 * (H - padT - padB);
    ctx.beginPath(); ctx.moveTo(padL, gy); ctx.lineTo(W - padR, gy); ctx.stroke();
    ctx.textAlign = "right"; ctx.fillText((ymax - i / 5 * (ymax - ymin)).toFixed(0), padL - 6, gy + 4);
  }
  ctx.textAlign = "center";
  ctx.fillText("Sensor reading (" + (plot.xlabel || "") + ")", (padL + W - padR) / 2, H - 6);

  series.forEach((ser, i) => {
    const col = SERIES_COLORS[i % SERIES_COLORS.length];
    ctx.fillStyle = col; ctx.strokeStyle = col;
    for (const p of ser.points) { ctx.beginPath(); ctx.arc(X(p[0]), Y(p[1]), 4, 0, Math.PI * 2); ctx.fill(); }
    if (ser.live) {
      if (s.mode === "RUN" && ser.targetY !== null && ser.targetY !== undefined) {
        ctx.setLineDash([5, 4]); ctx.beginPath();
        ctx.moveTo(X(ser.live[0]), Y(ser.live[1])); ctx.lineTo(X(ser.live[0]), Y(ser.targetY)); ctx.stroke();
        ctx.setLineDash([]);
      }
      drawStar(ctx, X(ser.live[0]), Y(ser.live[1]), 7);
    }
  });

  el("legend").innerHTML = series.map((ser, i) =>
    '<span style="color:' + SERIES_COLORS[i % SERIES_COLORS.length] + '">● ' + ser.label + '</span>').join("  &nbsp; ")
    + '  &nbsp; <span>★ live</span>';
}

function drawStar(ctx, cx, cy, r) {
  ctx.beginPath();
  for (let i = 0; i < 10; i++) {
    const rad = (i % 2 === 0) ? r : r * 0.45;
    const a = Math.PI / 5 * i - Math.PI / 2;
    const x = cx + Math.cos(a) * rad, y = cy + Math.sin(a) * rad;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.closePath(); ctx.fill();
}

async function poll() {
  try { S = await api("/api/state"); render(); } catch (e) { /* server busy; retry */ }
  setTimeout(poll, 100);
}
poll();
</script>
</body>
</html>
"""


# =============================================================================
# HTTP server
# =============================================================================
STATE = SmartMotorState()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # quiet

    def _send(self, code, body, ctype="application/json"):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj))

    def _body(self):
        length = int(self.headers.get("Content-Length", 0))
        if not length:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE, "text/html; charset=utf-8")
        elif self.path == "/api/state":
            self._json(STATE.snapshot())
        elif self.path == "/api/export":
            self._send(200, STATE.export_json(), "application/json")
        else:
            self._send(404, "not found", "text/plain")

    def do_POST(self):
        try:
            body = self._body()
            if self.path == "/api/config":
                STATE.set_config(body["backend"], body.get("motor"), body.get("sensor"))
            elif self.path == "/api/connect":
                STATE.connect(body)
            elif self.path == "/api/disconnect":
                STATE.disconnect()
            elif self.path == "/api/mode":
                STATE.set_mode(body["mode"])
            elif self.path == "/api/mapping":
                STATE.set_mapping(body["mode"])
            elif self.path == "/api/features":
                STATE.set_features(body["features"])
            elif self.path == "/api/axis_feature":
                STATE.set_axis_feature(int(body["axis"]), body["feature"])
            elif self.path == "/api/display":
                STATE.set_display_feature(body["feature"])
            elif self.path == "/api/record":
                STATE.record(axis=body.get("axis"))
            elif self.path == "/api/clear":
                STATE.clear_points(axis=body.get("axis"))
            elif self.path == "/api/sim":
                if "feature" in body:
                    STATE.sim_sensor(body["feature"], body["value"])
                if "motor" in body:
                    STATE.sim_motor(body["motor"])
            elif self.path == "/api/import":
                STATE.import_json(body["text"])
            else:
                self._send(404, "not found", "text/plain")
                return
            self._json({"ok": True})
        except Exception as exc:  # noqa: BLE001 — report errors to the browser
            self._json({"error": str(exc)}, code=400)


def main():
    parser = argparse.ArgumentParser(description="Smart Motor web interface")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"Smart Motor web interface running at {url}")
    print("Press Ctrl+C to stop.")
    if not HAS_LE:
        print("Note: 'legoeducation' is not installed — only the Simulated backend "
              "is available. Install it with: pip install legoeducation")
    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down…")
    finally:
        STATE.disconnect()
        server.server_close()


if __name__ == "__main__":
    main()
