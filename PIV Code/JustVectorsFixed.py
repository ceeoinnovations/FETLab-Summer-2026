# GENX320 Event Camera - Velocity Vector History
# Board:  OpenMV N6 + GENX320 Event Camera Module
# IDE:    OpenMV IDE
#
# Same acquisition + blob tracking as before, but instead of overlaying the
# current velocity vector on top of the live (noisy) event-histogram frame,
# each track keeps a short history of its own past velocity vectors. Every
# loop we redraw that whole history onto a blank canvas, fading older
# vectors out, so what you see is a trail showing how the vector evolved
# over the last N samples rather than just the latest one.
#
# The real sensor frame is still captured and used internally to find
# blobs - it's just not what gets displayed. Set SHOW_SOURCE_FRAME = True
# below if you want the raw frame back as a dim background reference.

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
csi0.framerate(50)

# csi0.ioctl(csi.IOCTL_GENX320_SET_BIASES, csi.GENX320_BIASES_DEFAULT)
# csi0.ioctl(csi.IOCTL_GENX320_SET_AFK, 1, 100, 150)

clock = time.clock()

# ---------------- Blob / tracking parameters ----------------
BLOB_THRESHOLDS = [(85, 95, -10, 10, -10, 10)]
BLOB_INVERT = True
PIXELS_THRESHOLD = 10
AREA_THRESHOLD = 100

MAX_MATCH_DIST = 40
MAX_MISSES = 5
SMOOTHING = 0.5

# ---------------- Vector history parameters ----------------
HISTORY_LEN = 15         # how many past vectors to keep/draw per track
MIN_SPEED_TO_RECORD = 2.0  # px/s - don't bother recording near-zero jitter
VECTOR_TIME = 0.2        # how far each drawn vector reaches (seconds of travel)
SHOW_SOURCE_FRAME = False  # True = draw the trail over a dim copy of the sensor frame

# Oldest -> newest color fade (dim gray -> bright cyan)
COLOR_OLD = (40, 40, 40)
COLOR_NEW = (0, 255, 255)

tracks = []   # dicts: id, cx, cy, vx, vy, t, misses, history
next_id = 0


def dist(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def lerp_color(c0, c1, f):
    return (int(c0[0] + (c1[0] - c0[0]) * f),
            int(c0[1] + (c1[1] - c0[1]) * f),
            int(c0[2] + (c1[2] - c0[2]) * f))


def draw_arrow(img, x0, y0, x1, y1, color, thickness=1, head_len=5):
    img.draw_line((int(x0), int(y0), int(x1), int(y1)),
                   color=color, thickness=thickness)
    dx, dy = x1 - x0, y1 - y0
    length = (dx * dx + dy * dy) ** 0.5
    if length < 1:
        return
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
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
    src = csi0.snapshot()
    now = time.ticks_us()

    blobs = src.find_blobs(BLOB_THRESHOLDS, invert=BLOB_INVERT,
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
            dt = time.ticks_diff(now, tr["t"]) / 1e6
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

            # Record this sample into the track's vector history.
            speed = (tr["vx"] ** 2 + tr["vy"] ** 2) ** 0.5
            if speed > MIN_SPEED_TO_RECORD:
                tr["history"].append((tr["cx"], tr["cy"], tr["vx"], tr["vy"]))
                if len(tr["history"]) > HISTORY_LEN:
                    tr["history"].pop(0)
        else:
            tr["misses"] += 1

    tracks = [tr for tr in tracks if tr["misses"] <= MAX_MISSES]

    for b in unmatched:
        tracks.append({"id": next_id, "cx": b.cx, "cy": b.cy,
                        "vx": 0.0, "vy": 0.0, "t": now, "misses": 0,
                        "history": []})
        next_id += 1

    # ---- build the display canvas ----
    if SHOW_SOURCE_FRAME:
        canvas = src
        canvas.b_and_w()  # dim it down so the trail stands out on top
    else:
        canvas = image.Image(src.width(), src.height(), image.RGB565)
        canvas.clear()

    for tr in tracks:
        hist = tr["history"]
        n = len(hist)
        for i, (hx, hy, hvx, hvy) in enumerate(hist):
            age_fraction = (i + 1) / n  # 0 = oldest, 1 = newest
            color = lerp_color(COLOR_OLD, COLOR_NEW, age_fraction)
            ex = hx + hvx * VECTOR_TIME
            ey = hy + hvy * VECTOR_TIME
            thickness = 1 if age_fraction < 0.7 else 2
            draw_arrow(canvas, hx, hy, ex, ey, color, thickness=thickness)

        # Mark the dot's current position and label its current speed.
        if tr["misses"] == 0:
            canvas.draw_cross((int(tr["cx"]), int(tr["cy"])), color=(255, 0, 0))
            speed = (tr["vx"] ** 2 + tr["vy"] ** 2) ** 0.5
            canvas.draw_string((int(tr["cx"]) + 6, int(tr["cy"]) - 10),
                                "%d px/s" % int(speed), color=(255, 255, 0))

    canvas.flush()
    print(clock.fps())
