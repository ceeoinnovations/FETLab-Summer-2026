'''
Camera selection helper for the driving project.

Primary path: automatically find and open Camo Studio's virtual camera by
NAME, with zero prompts and zero config editing. This works the same way
on Windows and macOS because it goes through cv2_enumerate_cameras, which
queries OpenCV's own backend device lists directly (AVFoundation on macOS,
DirectShow/Media Foundation on Windows) — the same lists cv2.VideoCapture
itself uses — so the name-to-index pairing it returns is guaranteed to
open the right device. (This sidesteps an earlier, less reliable approach
of cross-referencing a separate OS reporting tool like macOS's
system_profiler, whose ordering isn't guaranteed to match OpenCV's.)

Fallback path: if Camo can't be found automatically (not running yet,
ambiguous match, or the optional dependency isn't installed), this falls
back to the interactive thumbnail picker from before, or to the phone
stream URL configured in config.py (e.g. for IP Webcam, which streams
over a network URL rather than registering as a named local device, so
it can't be found this way).

Usage:
    from config import CAMERA
    from camlib import pick_camera
    cap = pick_camera(default_camera=CAMERA)
'''

import cv2
import numpy as np

THUMB_W, THUMB_H = 320, 240
CAMO_NAME_HINT = "camo"  # matched case-insensitively against the OS device name


def find_camo_camera():
    """
    Search all cameras the OS reports (via OpenCV's own backend
    enumeration) for one whose name contains "Camo". Returns
    (index, backend) if exactly one match is found so the caller can
    auto-select it with confidence. Returns None if zero or multiple
    matches are found (ambiguous — the caller should fall back to asking
    the user), or if cv2_enumerate_cameras isn't installed.
    """
    try:
        from cv2_enumerate_cameras import enumerate_cameras
    except ImportError:
        print("(Optional: `pip install cv2_enumerate_cameras` to auto-detect Camo Studio.)")
        return None

    print("Looking for Camo Studio...")
    try:
        cams = enumerate_cameras()
    except Exception as e:
        print(f"Camera enumeration failed ({e}) — falling back to manual selection.")
        return None

    if cams:
        print("Cameras detected:")
        for c in cams:
            print(f"   [{c.index}] {c.name!r}")
    else:
        print("No cameras detected at all by cv2_enumerate_cameras.")

    matches = [c for c in cams if CAMO_NAME_HINT in c.name.lower()]
    if len(matches) == 1:
        return matches[0].index, matches[0].backend
    if len(matches) > 1:
        print(f"Found {len(matches)} cameras with 'Camo' in the name — "
              "ambiguous, so falling back to manual selection.")
    elif cams:
        print("None of the detected cameras have 'Camo' in the name — "
              "falling back to manual selection.")
    return None


def _grab_preview(source):
    """Try to open `source` (an int index, an (index, backend) pair, or a
    URL string) and grab a warmed-up preview frame. Returns
    (frame_or_None, opened_bool)."""
    cap = cv2.VideoCapture(*source) if isinstance(source, tuple) else cv2.VideoCapture(source)
    if not cap.isOpened():
        cap.release()
        return None, False
    frame = None
    for _ in range(5):
        ret, f = cap.read()
        if ret and f is not None:
            frame = f
    cap.release()
    return frame, frame is not None


def pick_camera(default_camera=0, local_scan_range=4):
    """
    Try to find and open Camo Studio's camera automatically by name —
    no prompts, no config changes needed. If that fails (Camo not
    running, ambiguous match, or the optional dependency missing), fall
    back to the interactive thumbnail picker: local webcam indices plus
    whatever's configured in config.py (a URL for something like IP
    Webcam, or a local index for another device).

    Returns an opened cv2.VideoCapture.
    Raises RuntimeError if nothing at all can be opened.
    """
    found = find_camo_camera()
    if found is not None:
        index, backend = found
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            print(f"Found Camo Studio automatically (camera index {index}) — using it.\n")
            return cap
        cap.release()
        print("Found Camo Studio by name but couldn't open it — falling back to manual selection.")

    # ── Fallback: manual picker (unchanged from before) ─────────────────
    candidates = []  # (key_label, source, display_name, frame, ok)

    print(f"Scanning local camera indices 0-{local_scan_range - 1}...")
    for i in range(local_scan_range):
        frame, ok = _grab_preview(i)
        if ok:
            is_configured = (i == default_camera)
            name = f"Local camera {i}" + (" (configured)" if is_configured else "")
            candidates.append((str(i), i, name, frame, ok))

    # Only add a separate tile for the configured camera if it isn't
    # already one of the local indices above (e.g. it's a phone stream
    # URL, or a local index outside the scanned range).
    already_covered = any(source == default_camera for _, source, _, _, _ in candidates)
    if not already_covered:
        print(f"Checking configured camera (config.py): {default_camera}")
        frame, ok = _grab_preview(default_camera)
        candidates.append(('p', default_camera, f"Configured: {default_camera}", frame, ok))

    usable = [c for c in candidates if c[4]]
    if not usable:
        raise RuntimeError(
            f"Could not open any camera, including the configured one ({default_camera}).\n"
            "Check CAMERA in config.py and your Wi-Fi / USB connection."
        )

    if len(usable) == 1:
        _, source, name, _, _ = usable[0]
        print(f"Only one camera available — using it: {name}\n")
        chosen_source = source
    else:
        blank = np.zeros((THUMB_H, THUMB_W, 3), dtype=np.uint8)
        tiles, key_map = [], {}
        for key_label, source, name, frame, _ in usable:
            tile = cv2.resize(frame, (THUMB_W, THUMB_H)) if frame is not None else blank.copy()
            cv2.rectangle(tile, (0, 0), (THUMB_W, 34), (0, 0, 0), -1)
            cv2.putText(tile, name, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 80), 2)
            cv2.putText(tile, f"press  {key_label}  to select", (8, THUMB_H - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            tiles.append(tile)
            key_map[ord(key_label)] = source
            if key_label != 'p' and source == default_camera:
                key_map[ord('p')] = source  # 'p' also works as an alias for the configured camera

        grid = np.hstack(tiles)
        cv2.imshow("Select camera — press its key (Q/ESC = first option)", grid)

        chosen_source = usable[0][1]  # default to first if Q/ESC pressed
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key in key_map:
                chosen_source = key_map[key]
                break
            elif key in (27, ord('q')):
                break
        cv2.destroyAllWindows()

    print(f"Using camera: {chosen_source}\n")
    cap = cv2.VideoCapture(chosen_source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera: {chosen_source}")
    return cap