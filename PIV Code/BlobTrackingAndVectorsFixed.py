# GENX320 Event Camera - Multi-Dot Tracking with Velocity Vectors
# Board:  OpenMV N6 + GENX320 Event Camera Module
# IDE:    OpenMV IDE
#
# Acquisition is the plain csi0.snapshot() + color_palette approach (event
# histogram frames rendered as a false-color RGB565 image), same as
# OpenMV's own genx320 example. On top of that this script adds a
# lightweight nearest-neighbor tracker: each frame, blob detections are
# matched to existing tracks by closest distance, and velocity is computed
# from the change in position over elapsed time. An arrow + speed label is
# drawn on top of each tracked dot.

import csi
import image
import time

# ---------------- Sensor setup ----------------
csi0 = csi.CSI(cid=csi.GENX320)
csi0.reset()
csi0.pixformat(csi.GRAYSCALE)
csi0.framesize((320, 320))
csi0.brightness(128)  # Leave at 128 generally (this is the default).
csi0.contrast(16)  # Increase to make the image pop.
csi0.color_palette(image.PALETTE_EVT_LIGHT)  # image.PALETTE_EVT_DARK for dark mode.
# The default frame rate is 50 FPS. You can change it between ~20 FPS and ~350 FPS.
csi0.framerate(50)

# Bias preset - pick whichever matches your dots:
#   GENX320_BIASES_DEFAULT        - general purpose, reflective/high-contrast dots
#   GENX320_BIASES_LOW_LIGHT      - dim ambient lighting
#   GENX320_BIASES_ACTIVE_MARKER  - best if your "dots" are bright/IR LEDs
#   GENX320_BIASES_LOW_NOISE      - cleaner output, less sensitive
#   GENX320_BIASES_HIGH_SPEED     - very fast-moving dots, noisier output
# csi0.ioctl(csi.IOCTL_GENX320_SET_BIASES, csi.GENX320_BIASES_DEFAULT)

# Optional: notch out mains-frequency flicker if indoor lighting adds noise.
# csi0.ioctl(csi.IOCTL_GENX320_SET_AFK, 1, 100, 150)

clock = time.clock()

# ---------------- Tracking parameters (tune these) ----------------
# LAB-style 6-tuple threshold (L_min, L_max, A_min, A_max, B_min, B_max).
# With invert=True this selects anything that DEVIATS from the flat
# background gray that PALETTE_EVT_LIGHT/DARK renders when there's no event
# activity - i.e. the moving dots. Widen/narrow the L range to control
# sensitivity.
BLOB_THRESHOLDS = [(85, 95, -10, 10, -10, 10)]
BLOB_INVERT = True
PIXELS_THRESHOLD = 10
AREA_THRESHOLD = 100

MAX_MATCH_DIST = 40             # max px a dot may move between frames and
                                 # still be considered the same dot
MAX_MISSES = 5                  # frames a track may go undetected before
                                 # being dropped
MIN_SPEED_TO_DRAW = 2.0         # px/s - ignore jitter below this
VECTOR_TIME = 0.2               # arrow shows where the dot will be in this
                                 # many seconds at its current velocity
SMOOTHING = 0.5                 # 0 = no smoothing, closer to 1 = smoother/slower

tracks = []   # list of dicts: id, cx, cy, vx, vy, t (us), misses
next_id = 0


def dist(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def draw_arrow(img, x0, y0, x1, y1, color=(0, 255, 255), thickness=2, head_len=6):
    img.draw_line((int(x0), int(y0), int(x1), int(y1)),
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
    img.draw_line((int(x1), int(y1), int(left_x), int(left_y)),
                   color=color, thickness=thickness)
    img.draw_line((int(x1), int(y1), int(right_x), int(right_y)),
                   color=color, thickness=thickness)


while True:
    clock.tick()
    img = csi0.snapshot()
    # img.median(1)  # uncomment for noise cleanup.
    now = time.ticks_us()

    blobs = img.find_blobs(BLOB_THRESHOLDS, invert=BLOB_INVERT,
                            pixels_threshold=PIXELS_THRESHOLD,
                            area_threshold=AREA_THRESHOLD, merge=True)

    # ---- match detections to existing tracks (nearest neighbor) ----
    unmatched = list(blobs)
    for tr in tracks:
        best_blob = None
        best_dist = MAX_MATCH_DIST
        for b in unmatched:
            d = dist(tr["cx"], tr["cy"], b.cx, b.cy)
            if d < best_dist:
                best_dist = d
                best_blob = b

        if best_blob is not None:
            dt = time.ticks_diff(now, tr["t"]) / 1e6  # seconds
            if dt > 0:
                vx = (best_blob.cx - tr["cx"]) / dt
                vy = (best_blob.cy - tr["cy"]) / dt
                tr["vx"] = SMOOTHING * tr["vx"] + (1 - SMOOTHING) * vx
                tr["vy"] = SMOOTHING * tr["vy"] + (1 - SMOOTHING) * vy
            tr["cx"] = best_blob.cx
            tr["cy"] = best_blob.cy
            tr["t"] = now
            tr["misses"] = 0
            unmatched.remove(best_blob)
        else:
            tr["misses"] += 1

    tracks = [tr for tr in tracks if tr["misses"] <= MAX_MISSES]

    for b in unmatched:
        tracks.append({"id": next_id, "cx": b.cx, "cy": b.cy,
                        "vx": 0.0, "vy": 0.0, "t": now, "misses": 0})
        next_id += 1

    # ---- draw ----
    for b in blobs:
        img.draw_rectangle(b.rect, color=(255, 0, 0))

    for tr in tracks:
        if tr["misses"] > 0:
            continue
        cx, cy = tr["cx"], tr["cy"]
        speed = (tr["vx"] ** 2 + tr["vy"] ** 2) ** 0.5
        img.draw_cross((int(cx), int(cy)), color=(0, 255, 0))
        if speed > MIN_SPEED_TO_DRAW:
            ex = cx + tr["vx"] * VECTOR_TIME
            ey = cy + tr["vy"] * VECTOR_TIME
            draw_arrow(img, cx, cy, ex, ey, color=(0, 255, 255), thickness=2)
            img.draw_string((int(cx) + 6, int(cy) - 10),
                             "%d px/s" % int(speed), color=(255, 255, 0))

    print(clock.fps())
