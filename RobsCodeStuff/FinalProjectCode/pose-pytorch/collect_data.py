"""
Step 1 — Collect training images.

Controls:
  1-5     select gesture class
  Space   start a timed capture burst (auto-captures at TIMER_CAPTURE_HZ
          for TIMER_DURATION_SEC seconds, then stops on its own)
  Enter   tap once = capture a single frame right now
          hold down = keeps capturing frames for as long as it's held
  Q       quit

Adjustable settings (below): TIMER_CAPTURE_HZ, TIMER_DURATION_SEC,
ENTER_MIN_INTERVAL_SEC.
"""

import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
from model import GESTURE_CLASSES
from camlib import pick_camera

# ── Adjustable capture settings ──────────────────────────────────────────
TIMER_CAPTURE_HZ      = 10   # frames captured per second during a Space burst
TIMER_DURATION_SEC    = 3.0  # how long a Space burst lasts, in seconds
ENTER_MIN_INTERVAL_SEC = 0.15 # minimum seconds between Enter captures.
                              # Holding Enter relies on your OS/keyboard's
                              # key-repeat feature to keep sending the key —
                              # this value throttles how fast that turns into
                              # saved frames, so you don't flood near-duplicate
                              # images if your OS repeats keys very quickly.
# ──────────────────────────────────────────────────────────────────────────

TIMER_CAPTURE_INTERVAL = 1.0 / TIMER_CAPTURE_HZ
ENTER_KEYS = (13, 10)  # Enter/Return show up as 13 (CR) or 10 (LF) depending on OS

DATA_DIR = Path(__file__).parent / "data"
for cls in GESTURE_CLASSES:
    (DATA_DIR / cls).mkdir(parents=True, exist_ok=True)

counts  = {cls: len(list((DATA_DIR / cls).iterdir())) for cls in GESTURE_CLASSES}
KEY_MAP = {ord(str(i + 1)): GESTURE_CLASSES[i] for i in range(len(GESTURE_CLASSES))}

cap, _        = pick_camera()
current_class = None

# Timed-burst state
recording        = False
recording_start  = 0.0
last_timed_capture = 0.0

# Enter-key throttle state
last_enter_capture = 0.0

print("Keys: 1-5 select class | Space = timed burst | Enter = tap/hold to capture | Q = quit")
for i, cls in enumerate(GESTURE_CLASSES):
    print(f"  {i + 1} = {cls}")
print(f"Timed burst: {TIMER_CAPTURE_HZ} Hz for {TIMER_DURATION_SEC}s")


def save_frame(cls, frame):
    path = str(DATA_DIR / cls / f"{counts[cls]:05d}.jpg")
    cv2.imwrite(path, frame)  # save the clean, un-overlaid frame
    counts[cls] += 1
    print(f"Saved {path}")


def draw_readable_text(img, text, org, font_scale, color, thickness):
    """Draw text over a thin translucent dark bar so it stays legible
    against busy/bright backgrounds, without a big opaque block."""
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x, y = org
    pad = 4
    overlay = img.copy()
    cv2.rectangle(overlay, (x - pad, y - th - pad), (x + tw + pad, y + baseline + pad),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    now = time.time()

    # Draw the UI on a copy so the overlay never touches the raw frame
    # that gets saved to disk — the model should only ever see the pose.
    display = frame.copy()

    label = current_class or "— press 1-5 to select a class"
    draw_readable_text(display, f"Class: {label}", (10, 32), 0.9, (0, 255, 80), 2)
    for i, cls in enumerate(GESTURE_CLASSES):
        draw_readable_text(display, f"  {i + 1}: {cls}  ({counts[cls]} saved)",
                            (10, 62 + i * 26), 0.55, (220, 220, 220), 1)
    draw_readable_text(display, "Space = timed burst  |  Enter = tap/hold  |  Q = quit",
                        (10, display.shape[0] - 10), 0.5, (235, 235, 235), 1)

    if recording:
        remaining = TIMER_DURATION_SEC - (now - recording_start)
        draw_readable_text(display, f"RECORDING  {max(remaining, 0):.1f}s left",
                            (10, display.shape[0] - 36), 0.62, (60, 60, 255), 2)

    cv2.imshow("Collect Data", display)
    key = cv2.waitKey(1) & 0xFF

    # ── Handle a running timed burst ────────────────────────────────────
    if recording:
        if now - recording_start >= TIMER_DURATION_SEC:
            recording = False
            print("Timed burst finished.")
        elif now - last_timed_capture >= TIMER_CAPTURE_INTERVAL:
            save_frame(current_class, frame)
            last_timed_capture = now

    if key == ord('q'):
        break
    elif key in KEY_MAP:
        if not recording:  # don't let class change mid-burst
            current_class = KEY_MAP[key]
            print(f"Selected: {current_class}")
    elif key == ord(' ') and current_class and not recording:
        recording = True
        recording_start = now
        last_timed_capture = 0.0  # capture the first frame immediately
        print(f"Starting timed burst for '{current_class}'...")
    elif key in ENTER_KEYS and current_class and not recording:
        if now - last_enter_capture >= ENTER_MIN_INTERVAL_SEC:
            save_frame(current_class, frame)
            last_enter_capture = now

cap.release()
cv2.destroyAllWindows()
print("\nCollection complete. Image counts:", counts)