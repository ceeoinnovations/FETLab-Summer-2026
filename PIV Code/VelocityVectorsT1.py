#RUN in the OpenMV IDE
#Untitled - By: vanes - Wed Jul 1 2026
# GENX320 Event Camera - Multi-Dot Tracking with Velocity Vectors
# Board:  OpenMV N6 + GENX320 Event Camera Module
# IDE:    OpenMV IDE  (requires firmware with csi + GENX320 event mode)
#
# How it works:
#   In event mode the GENX320 hands you a raw stream of contrast-detection
#   events: (type, s, ms, us, x, y). There's no ready-made frame - you build
#   one yourself with img.draw_event_histogram(), which paints each event
#   into a 320x320 grayscale image around a baseline "brightness" value.
#   A dot that's moving keeps generating events, so it shows up as a blob
#   that deviates from the baseline (brighter OR darker, depending on
#   contrast polarity); static background stays at the baseline.
#
#   Blob detection alone doesn't preserve identity across frames, so this
#   script layers a lightweight nearest-neighbor tracker on top: each loop,
#   new blob detections are matched to existing tracks by closest distance,
#   and velocity is computed from the change in position over elapsed time.
#   An arrow + speed label is drawn on top of each tracked dot.

import csi
import image
import time
from ulab import numpy as np

# ---------------- Event camera setup ----------------
EVENT_BUF_SIZE = 2048  # must be a power of two, 1024-65536

# Surface we draw the event histogram into every loop.
img = image.Image(320, 320, image.GRAYSCALE)

# Event buffer: EVENT_BUF_SIZE rows x 6 columns
#   [0] event type, [1] s, [2] ms, [3] us, [4] x (0-319), [5] y (0-319)
events = np.zeros((EVENT_BUF_SIZE, 6), dtype=np.uint16)

csi0 = csi.CSI(cid=csi.GENX320)
csi0.reset()
csi0.ioctl(csi.IOCTL_GENX320_SET_MODE, csi.GENX320_MODE_EVENT, EVENT_BUF_SIZE)

# Bias preset - pick whichever matches your dots:
#   GENX320_BIASES_DEFAULT        - general purpose, reflective/high-contrast dots
#   GENX320_BIASES_LOW_LIGHT      - dim ambient lighting
#   GENX320_BIASES_ACTIVE_MARKER  - best if your "dots" are bright/IR LEDs
#   GENX320_BIASES_LOW_NOISE      - cleaner output, less sensitive
#   GENX320_BIASES_HIGH_SPEED     - very fast-moving dots, noisier output
csi0.ioctl(csi.IOCTL_GENX320_SET_BIASES, csi.GENX320_BIASES_DEFAULT)

# Optional: notch out mains-frequency flicker (uncomment + tune if indoor
# lighting is adding noise, e.g. 100-120Hz covers most mains hum)
# csi0.ioctl(csi.IOCTL_GENX320_SET_AFK, 1, 100, 150)

clock = time.clock()

# ---------------- Histogram rendering parameters ----------------
BRIGHTNESS = 128   # baseline gray level with no events
CONTRAST = 64      # how much each event nudges a pixel away from baseline

# Blobs are detected as pixels that deviate from BRIGHTNESS in either
# direction (event activity), so we invert a band centered on it.
BLOB_THRESHOLDS = [(BRIGHTNESS - 20, BRIGHTNESS + 20)]
BLOB_INVERT = True

# ---------------- Tracking parameters (tune these) ----------------
PIXELS_THRESHOLD = 2
AREA_THRESHOLD = 4
MAX_MATCH_DIST = 40             # max px a dot may move between loops and
                                 # still be considered the same dot
MAX_MISSES = 5                  # loops a track may go undetected before
                                 # being dropped
MIN_SPEED_TO_DRAW = 2.0         # px/s - ignore jitter below this
VECTOR_TIME = 0.2               # arrow shows where the dot will be in this
                                 # many seconds at its current velocity
SMOOTHING = 0.5                 # 0 = no smoothing, closer to 1 = smoother/slower

tracks = []   # list of dicts: id, cx, cy, vx, vy, t (us), misses
next_id = 0


def dist(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def draw_arrow(img, x0, y0, x1, y1, color=255, thickness=2, head_len=6):
    img.draw_line(int(x0), int(y0), int(x1), int(y1),
                   color=color, thickness=thickness)
    dx, dy = x1 - x0, y1 - y0
    length = (dx * dx + dy * dy) ** 0.5
    if length < 1:
        return
    ux, uy = dx / length, dy / length
    px, py = -uy, ux  # perpendicular
    back_x, back_y = x1 - ux * head_len, y1 - uy * head_len
    left_x = back_x + px * (head_len * 0.5)
    left_y = back_y + py * (head_len * 0.5)
    right_x = back_x - px * (head_len * 0.5)
    right_y = back_y - py * (head_len * 0.5)
    img.draw_line(int(x1), int(y1), int(left_x), int(left_y),
                   color=color, thickness=thickness)
    img.draw_line(int(x1), int(y1), int(right_x), int(right_y),
                   color=color, thickness=thickness)


while True:
    clock.tick()

    # Pull up to EVENT_BUF_SIZE events off the sensor (old buffer contents
    # aren't cleared beforehand, so event_count tells you how many are valid).
    event_count = csi0.ioctl(csi.IOCTL_GENX320_READ_EVENTS, events)
    now = time.ticks_us()

    # Render this batch of events into a fresh histogram frame.
    img.draw_event_histogram(events[:event_count], clear=True,
                              brightness=BRIGHTNESS, contrast=CONTRAST)

    blobs = img.find_blobs(BLOB_THRESHOLDS, invert=BLOB_INVERT,
                            pixels_threshold=PIXELS_THRESHOLD,
                            area_threshold=AREA_THRESHOLD, merge=True)

    # ---- match detections to existing tracks (nearest neighbor) ----
    unmatched = list(blobs)
    for tr in tracks:
        best_blob = None
        best_dist = MAX_MATCH_DIST
        for b in unmatched:
            d = dist(tr["cx"], tr["cy"], b.cx(), b.cy())
            if d < best_dist:
                best_dist = d
                best_blob = b

        if best_blob is not None:
            dt = time.ticks_diff(now, tr["t"]) / 1e6  # seconds
            if dt > 0:
                vx = (best_blob.cx() - tr["cx"]) / dt
                vy = (best_blob.cy() - tr["cy"]) / dt
                tr["vx"] = SMOOTHING * tr["vx"] + (1 - SMOOTHING) * vx
                tr["vy"] = SMOOTHING * tr["vy"] + (1 - SMOOTHING) * vy
            tr["cx"] = best_blob.cx()
            tr["cy"] = best_blob.cy()
            tr["t"] = now
            tr["misses"] = 0
            unmatched.remove(best_blob)
        else:
            tr["misses"] += 1

    tracks = [tr for tr in tracks if tr["misses"] <= MAX_MISSES]

    for b in unmatched:
        tracks.append({"id": next_id, "cx": b.cx(), "cy": b.cy(),
                        "vx": 0.0, "vy": 0.0, "t": now, "misses": 0})
        next_id += 1

    # ---- draw ----
    for b in blobs:
        img.draw_rectangle(b.rect(), color=255)

    for tr in tracks:
        if tr["misses"] > 0:
            continue
        cx, cy = tr["cx"], tr["cy"]
        speed = (tr["vx"] ** 2 + tr["vy"] ** 2) ** 0.5
        img.draw_cross(int(cx), int(cy), color=255, size=4)
        if speed > MIN_SPEED_TO_DRAW:
            ex = cx + tr["vx"] * VECTOR_TIME
            ey = cy + tr["vy"] * VECTOR_TIME
            draw_arrow(img, cx, cy, ex, ey, color=255, thickness=2)
            img.draw_string(int(cx) + 6, int(cy) - 10,
                             "%d px/s" % int(speed), color=255)

    # Push the annotated image to the IDE's frame buffer.
    img.flush()
    print(event_count, clock.fps())
