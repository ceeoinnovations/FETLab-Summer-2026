# GENX320 Event Camera - Raw Event Logger
# Board: OpenMV N6 + GENX320 Event Camera Module
# IDE:   OpenMV IDE
#
# Captures the raw (type, s, ms, us, x, y) event stream straight off the
# GENX320 and writes it to a binary log file for offline processing on a
# host PC - e.g. patch-wise contrast-maximization / motion-compensation
# velocimetry, or any other event-level analysis that a histogram/blob
# pipeline would throw away.
#
# FILE FORMAT
# -----------
# No header, no delimiters - just a flat run of little-endian uint16
# values, 6 per event, back to back:
#   [0] event type   (0 = OFF/negative contrast, 1 = ON/positive contrast)
#   [1] seconds
#   [2] milliseconds (0-999, sub-second)
#   [3] microseconds (0-999, sub-millisecond)
#   [4] x (0-319)
#   [5] y (0-319)
#
# Reconstruct an absolute microsecond timestamp on the host with:
#   t_us = s * 1_000_000 + ms * 1000 + us
#
# To load it back in Python:
#   import numpy as np
#   events = np.fromfile("events.bin", dtype="<u2").reshape(-1, 6)
#   ev_type, s, ms, us, x, y = events.T
#   t_us = s.astype(np.int64) * 1_000_000 + ms * 1000 + us

import csi
import time
import uos
from ulab import numpy as np

# ---------------- Event camera setup ----------------
EVENT_BUF_SIZE = 4096  # power of two, 1024-65536. Bigger = fewer read calls
                        # per loop, but more risk of a read lagging behind
                        # the true event rate if your loop can't keep up.

events = np.zeros((EVENT_BUF_SIZE, 6), dtype=np.uint16)

csi0 = csi.CSI(cid=csi.GENX320)
csi0.reset()
csi0.ioctl(csi.IOCTL_GENX320_SET_MODE, csi.GENX320_MODE_EVENT, EVENT_BUF_SIZE)

# Bias preset - pick whichever matches your tracer particles/dots:
#   GENX320_BIASES_DEFAULT        - general purpose
#   GENX320_BIASES_LOW_LIGHT      - dim ambient lighting
#   GENX320_BIASES_ACTIVE_MARKER  - best for bright/IR LED markers
#   GENX320_BIASES_LOW_NOISE      - cleaner output, less sensitive (good
#                                    starting point for velocimetry - less
#                                    background noise polluting the IWE)
#   GENX320_BIASES_HIGH_SPEED     - very fast motion, noisier output
csi0.ioctl(csi.IOCTL_GENX320_SET_BIASES, csi.GENX320_BIASES_DEFAULT)

# Optional: notch out mains-frequency flicker if indoor lighting adds noise.
# csi0.ioctl(csi.IOCTL_GENX320_SET_AFK, 1, 100, 150)

# ---------------- Logging setup ----------------
LOG_PATH = "/sdcard/events.bin"  # change to "/flash/events.bin" if no SD card
RECORD_SECONDS = 10              # set to None to record until you stop the script
FLUSH_EVERY_N_READS = 5          # write-through to disk every N buffer reads

# Wipe any previous log at this path so runs don't append to old data.
try:
    uos.remove(LOG_PATH)
except OSError:
    pass

clock = time.clock()
start_time = time.ticks_ms()
reads_since_flush = 0
total_events_logged = 0

with open(LOG_PATH, "wb") as f:
    while True:
        clock.tick()

        # Reads up to EVENT_BUF_SIZE events. Old buffer contents aren't
        # cleared beforehand, so event_count tells you how many rows in
        # `events` are actually valid this call.
        event_count = csi0.ioctl(csi.IOCTL_GENX320_READ_EVENTS, events)

        if event_count > 0:
            f.write(events[:event_count].tobytes())
            total_events_logged += event_count
            reads_since_flush += 1

            if reads_since_flush >= FLUSH_EVERY_N_READS:
                f.flush()
                reads_since_flush = 0

        elapsed_s = time.ticks_diff(time.ticks_ms(), start_time) / 1000
        print("events: %d  total: %d  fps: %.1f  t: %.1fs" %
              (event_count, total_events_logged, clock.fps(), elapsed_s))

        if RECORD_SECONDS is not None and elapsed_s >= RECORD_SECONDS:
            f.flush()
            break

print("Done. Logged %d events to %s" % (total_events_logged, LOG_PATH))
