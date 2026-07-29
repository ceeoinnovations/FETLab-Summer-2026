"""
Brick Sorting Machine (Color + Shape)
======================================
For the LEGO Education Computer Science & AI Kit Color Sensor, plus a laptop
webcam for shape detection, plus an optional Single Motor sorting arm.

Activity this implements:
  "Create different bins with labelled bricks (by colors/shapes) put in each,
   train a sorting model such that given a new brick the system can sort it
   to the corresponding bin."

Why two sensors?
----------------
The kit's Color Sensor reads color/reflection, but has no way to sense a
brick's shape. So shape detection comes from a webcam + simple image
analysis (contour/edge detection), while color still comes from the
official Color Sensor. Each trained bin gets a combined fingerprint:
  - COLOR:  normalized (r, g, b) proportions from the Color Sensor
  - SHAPE:  (circularity, aspect_ratio, extent) from the webcam image
A new brick is sorted into whichever trained bin's fingerprint is closest
(nearest-centroid classification), using both cues together.

Setup
-----
1. Install dependencies:
       pip install legoeducation opencv-python --break-system-packages
2. Update CARD_COLOR / CARD_SERIAL below to match the Connection Card that
   came with your Color Sensor.
3. Mount your webcam looking straight down at a plain, high-contrast
   surface (e.g. a sheet of white or black paper) where bricks will be
   placed. Good, even lighting with no strong shadows works best.
4. (Optional) If you've built a rotating sorting arm driven by a Single
   Motor, set USE_SORTING_ARM = True and fill in the motor's connection
   card info + one absolute position (in degrees) per bin.
5. Run:  python brick_sorter.py

Notes on shape detection
-------------------------
This uses classic computer vision (contours), not a trained neural network,
so it works instantly with no training images needed -- but it can only
distinguish coarse shape properties: how round vs. rectangular a brick's
silhouette is, and how elongated it is. It reads the brick's outline from
directly above, so it can't see 3D height differences (e.g. a "plate" vs
a taller "brick" of the same footprint will look identical to the camera).
If you need to reliably tell those apart, a trained image classifier
(Teachable Machine / Edge Impulse, like your Maze Lab project) would do
better -- happy to build that version instead if this turns out to be
the limiting factor for your bins.

Tested against the LEGO/LEGOEducation Python API docs and OpenCV 4.x as of
July 2026. If `import legoeducation` fails, run `pip install legoeducation
--upgrade --break-system-packages` -- the API is still young and evolving.
"""

import json
import math
import os
import time

import cv2
import legoeducation as le

# ---------------------------------------------------------------------------
# CONFIGURATION -- edit these to match your hardware
# ---------------------------------------------------------------------------

# Color Sensor's Connection Card (the little card that snaps onto the sensor)
CARD_COLOR = le.LEGO_COLOR_PURPLE
CARD_SERIAL = "6040"  # <-- change to match YOUR card's serial number

# Webcam
CAMERA_INDEX = 0            # 0 is usually the built-in/first webcam
MIN_CONTOUR_AREA = 800      # ignore specks smaller than this (pixels^2)
SHAPE_SAMPLES_PER_CAPTURE = 10
SHAPE_SAMPLE_DELAY = 0.1

# Set to True only if you've built a physical sorting arm with a Single Motor
USE_SORTING_ARM = False
ARM_CARD_COLOR = le.LEGO_COLOR_AZURE
ARM_CARD_SERIAL = "0000"  # <-- change to match your motor's Connection Card

# Map each bin label to an absolute position (degrees) for the sorting arm.
# e.g. ARM_POSITIONS = {"Red-Round": 0, "Blue-Long": 90, "Yellow-Square": 180}
ARM_POSITIONS = {}
ARM_HOME_POSITION = 0

MODEL_FILE = "brick_sorter_model.json"
COLOR_SAMPLES_PER_TRAINING = 12
COLOR_SAMPLE_DELAY = 0.15

# How much each cue counts toward the final decision. Raise SHAPE_WEIGHT if
# bins differ mainly by shape; raise COLOR_WEIGHT if they differ mainly by
# color. Setting either to 0 ignores that cue entirely.
COLOR_WEIGHT = 1.0
SHAPE_WEIGHT = 1.0


# ---------------------------------------------------------------------------
# Color features (from the LEGO Color Sensor)
# ---------------------------------------------------------------------------

def normalize_rgb(r, g, b):
    """Brightness-independent color proportions (r, g, b sum to 1), so
    holding a brick closer/farther from the sensor doesn't change its
    fingerprint -- only the color ratio matters."""
    total = r + g + b
    if total == 0:
        return (0.0, 0.0, 0.0)
    return (r / total, g / total, b / total)


def read_average_color(sensor, num_samples, delay):
    readings = []
    for i in range(num_samples):
        r, g, b = sensor.sensor.rawRed, sensor.sensor.rawGreen, sensor.sensor.rawBlue
        readings.append((r, g, b))
        print(f"  color reading {i + 1}/{num_samples}: raw=({r}, {g}, {b})")
        time.sleep(delay)
    avg = tuple(sum(x[i] for x in readings) / len(readings) for i in range(3))
    return normalize_rgb(*avg)


# ---------------------------------------------------------------------------
# Shape features (from the webcam)
# ---------------------------------------------------------------------------

def extract_shape_features(frame, min_area=MIN_CONTOUR_AREA):
    """Find the brick's silhouette and return (circularity, aspect_ratio,
    extent). Tries both bright-on-dark and dark-on-bright thresholding
    since we don't know in advance whether the brick or the background
    surface is lighter."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    frame_area = frame.shape[0] * frame.shape[1]

    best_contour, best_area = None, 0
    for flag in (cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV):
        _, thresh = cv2.threshold(blur, 0, 255, flag + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            # skip specks and skip blobs that are basically the whole frame
            # (that's the background surface, not the brick)
            if area < min_area or area > 0.85 * frame_area:
                continue
            if area > best_area:
                best_area, best_contour = area, c

    if best_contour is None:
        return None

    c = best_contour
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    circularity = (4 * math.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0
    x, y, w, h = cv2.boundingRect(c)
    ar = (w / h) if h > 0 else 0.0
    ar_norm = min(ar, 1 / ar) if ar > 0 else 0.0  # 0..1, 1 = perfectly square bbox
    extent = (area / (w * h)) if (w * h) > 0 else 0.0
    return circularity, ar_norm, extent, best_contour


def open_camera():
    """Opens the webcam ONCE for the whole program run. Reopening the camera
    for every single capture is unreliable on macOS (the camera backend can
    freeze or return stale frames on subsequent opens), so we keep one
    handle alive and reuse it everywhere."""
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Could not open the webcam. Check CAMERA_INDEX and that no")
        print("other program is using the camera.")
        return None
    # Let the camera warm up and flush the first few (often dark/stale) frames
    time.sleep(0.5)
    for _ in range(5):
        cap.read()
    return cap


def capture_average_shape(cap, num_samples, delay):
    """Shows a live preview window using the given (already-open) camera
    handle. Press SPACE to capture, ESC to cancel."""
    if cap is None or not cap.isOpened():
        print("Camera is not available.")
        return None

    print("Position the brick in view of the camera.")
    print("Press SPACE to capture, ESC to cancel.")
    features = []
    consecutive_failures = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                consecutive_failures += 1
                if consecutive_failures > 60:
                    print("Camera stopped responding -- try again, or check CAMERA_INDEX.")
                    return None
                cv2.waitKey(1)
                continue
            consecutive_failures = 0
            result = extract_shape_features(frame)
            display = frame.copy()
            if result:
                circularity, ar_norm, extent, contour = result
                cv2.drawContours(display, [contour], -1, (0, 255, 0), 2)
                cv2.putText(
                    display,
                    f"circularity={circularity:.2f} aspect={ar_norm:.2f} extent={extent:.2f}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2,
                )
            else:
                cv2.putText(display, "No brick detected", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow("Brick Shape Camera (SPACE = capture, ESC = cancel)", display)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            if key == 32:  # SPACE
                for i in range(num_samples):
                    ok, frame = cap.read()
                    if not ok:
                        cv2.waitKey(1)
                        time.sleep(delay)
                        continue
                    result = extract_shape_features(frame)
                    if result:
                        circularity, ar_norm, extent, _ = result
                        features.append((circularity, ar_norm, extent))
                        print(f"  shape reading {i + 1}/{num_samples}: "
                              f"circularity={circularity:.3f} aspect={ar_norm:.3f} extent={extent:.3f}")
                    time.sleep(delay)
                break
    finally:
        # Do NOT release cap here -- it's shared across the whole program.
        # Just close the preview window (and give macOS a tick to actually
        # process the close event, or it can leave the window in limbo).
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    if not features:
        print("No brick was detected during capture -- try again with better lighting/contrast.")
        return None
    return tuple(sum(f[i] for f in features) / len(features) for i in range(3))


# ---------------------------------------------------------------------------
# Model: bin label -> {"color": [r,g,b], "shape": [circularity, aspect, extent]}
# ---------------------------------------------------------------------------

def euclidean(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def load_model():
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, "r") as f:
            return json.load(f)
    return {}


def save_model(model):
    with open(MODEL_FILE, "w") as f:
        json.dump(model, f, indent=2)
    print(f"Model saved to {MODEL_FILE}")


def classify(color_sample, shape_sample, model):
    if not model:
        return None, None
    best_label, best_dist = None, float("inf")
    for label, fingerprint in model.items():
        color_dist = euclidean(color_sample, fingerprint["color"]) if color_sample else 0
        shape_dist = euclidean(shape_sample, fingerprint["shape"]) if shape_sample else 0
        total = COLOR_WEIGHT * color_dist + SHAPE_WEIGHT * shape_dist
        if total < best_dist:
            best_label, best_dist = label, total
    return best_label, best_dist


# ---------------------------------------------------------------------------
# Hardware connection helpers
# ---------------------------------------------------------------------------

def connect_color_sensor():
    sensor = le.ColorSensor()
    sensor.connect(card_color=CARD_COLOR, card_serial=CARD_SERIAL)
    if not sensor.connected:
        print("Could not connect to the Color Sensor. Check that it's")
        print("powered on, charged, and CARD_COLOR/CARD_SERIAL match.")
        raise SystemExit(1)
    print("Color Sensor connected.")
    return sensor


def connect_sorting_arm():
    motor = le.SingleMotor()
    motor.connect(card_color=ARM_CARD_COLOR, card_serial=ARM_CARD_SERIAL)
    if not motor.connected:
        print("Could not connect to the sorting arm motor. Continuing")
        print("without physical sorting -- results will just print to screen.")
        return None
    print("Sorting arm motor connected.")
    return motor


def route_to_bin(motor, label):
    if motor is None:
        return
    if label not in ARM_POSITIONS:
        print(f"(No arm position configured for bin '{label}' -- skipping move.)")
        return
    motor.motor_run_to_absolute_position(
        ARM_POSITIONS[label], direction=le.MOTOR_MOVE_DIRECTION_SHORTEST, speed=60
    )
    time.sleep(1.0)  # give the brick time to drop into the bin
    motor.motor_run_to_absolute_position(
        ARM_HOME_POSITION, direction=le.MOTOR_MOVE_DIRECTION_SHORTEST, speed=60
    )


# ---------------------------------------------------------------------------
# Menu actions
# ---------------------------------------------------------------------------

def do_train(sensor, model, cam):
    label = input("Bin label (e.g. Red-Round, Blue-Long, Yellow-Square): ").strip()
    if not label:
        print("No label entered, cancelling.")
        return

    input(f"Hold a brick from the '{label}' bin under the Color Sensor, "
          f"then press Enter to capture {COLOR_SAMPLES_PER_TRAINING} color readings...")
    color_fp = read_average_color(sensor, COLOR_SAMPLES_PER_TRAINING, COLOR_SAMPLE_DELAY)

    print("Now show the same brick to the webcam for shape capture.")
    shape_fp = capture_average_shape(cam, SHAPE_SAMPLES_PER_CAPTURE, SHAPE_SAMPLE_DELAY)
    if shape_fp is None:
        print("Shape capture failed/cancelled -- bin not saved. Try again.")
        return

    model[label] = {"color": list(color_fp), "shape": list(shape_fp)}
    print(f"Trained bin '{label}':")
    print(f"  color fingerprint (r,g,b) = {tuple(round(c, 3) for c in color_fp)}")
    print(f"  shape fingerprint (circularity, aspect, extent) = {tuple(round(s, 3) for s in shape_fp)}")
    save_model(model)


def do_sort(sensor, model, motor, cam):
    if not model:
        print("No bins trained yet! Train at least one bin first.")
        return

    input("Hold the new brick under the Color Sensor, then press Enter...")
    color_sample = read_average_color(sensor, 5, COLOR_SAMPLE_DELAY)

    print("Now show the same brick to the webcam.")
    shape_sample = capture_average_shape(cam, 5, SHAPE_SAMPLE_DELAY)
    if shape_sample is None:
        print("Shape capture failed/cancelled -- try again.")
        return

    label, dist = classify(color_sample, shape_sample, model)
    print(f"\n--> Sorted into bin: {label}  (distance={dist:.4f})\n")
    route_to_bin(motor, label)


def do_list(model):
    if not model:
        print("No bins trained yet.")
        return
    print("Trained bins:")
    for label, fp in model.items():
        print(f"  {label}:")
        print(f"    color = {tuple(round(c, 3) for c in fp['color'])}")
        print(f"    shape = {tuple(round(s, 3) for s in fp['shape'])}")


def do_delete(model):
    label = input("Bin label to delete: ").strip()
    if label in model:
        del model[label]
        save_model(model)
        print(f"Deleted bin '{label}'.")
    else:
        print("No such bin.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== Brick Sorting Machine (Color + Shape) ===\n")
    sensor = connect_color_sensor()
    motor = connect_sorting_arm() if USE_SORTING_ARM else None
    cam = open_camera()
    if cam is None:
        print("Continuing without shape detection -- shape captures will fail.")
    model = load_model()
    if model:
        print(f"Loaded existing model with bins: {list(model.keys())}")

    try:
        while True:
            print(
                "\nWhat would you like to do?\n"
                "  [1] Train a new bin (color + shape)\n"
                "  [2] Sort a new brick\n"
                "  [3] List trained bins\n"
                "  [4] Delete a bin\n"
                "  [5] Quit"
            )
            choice = input("> ").strip()
            if choice == "1":
                do_train(sensor, model, cam)
            elif choice == "2":
                do_sort(sensor, model, motor, cam)
            elif choice == "3":
                do_list(model)
            elif choice == "4":
                do_delete(model)
            elif choice == "5":
                break
            else:
                print("Please enter 1-5.")
    finally:
        sensor.disconnect()
        if motor is not None:
            motor.disconnect()
        if cam is not None:
            cam.release()
        cv2.destroyAllWindows()
        print("Disconnected. Goodbye!")


if __name__ == "__main__":
    main()