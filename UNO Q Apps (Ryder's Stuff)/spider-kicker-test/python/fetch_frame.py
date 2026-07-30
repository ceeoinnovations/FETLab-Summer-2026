"""Grab one JPEG frame from the running app's own /video_feed MJPEG stream,
for reading off (u, v) calibration points into points.csv - an alternative to
snapshot.py that needs the app STOPPED and its own environment (arduino
package, cv2). This script instead runs against the app WHILE IT'S RUNNING
(autonomous off is fine - no motion risk) using nothing but the stdlib, so it
works from any plain `python3`/venv, no special environment required.

    python3 fetch_frame.py                       # http://localhost:7000/video_feed
    python3 fetch_frame.py my_frame.jpg
    python3 fetch_frame.py my_frame.jpg http://<device-ip>:7000/video_feed

Note: the saved frame includes main.py's overlay text (phase/target, drawn in
the top-left corner) since it's baked into the JPEG the app publishes - avoid
picking calibration points from that corner, or temporarily ignore it.
"""
import sys
import urllib.request

DEFAULT_URL = "http://localhost:7000/video_feed"
BOUNDARY = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
JPEG_SOI = b"\xff\xd8"
JPEG_EOI = b"\xff\xd9"


def fetch_one_frame(url, timeout=10):
    """Read the multipart stream just long enough to pull out one complete
    JPEG (from its first SOI marker after a boundary to the matching EOI)."""
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        buf = b""
        while True:
            chunk = resp.read(8192)
            if not chunk:
                raise RuntimeError("Stream ended before a full frame was read.")
            buf += chunk

            start = buf.find(BOUNDARY)
            if start == -1:
                # keep only enough trailing bytes to catch a boundary split
                # across a chunk boundary
                buf = buf[-len(BOUNDARY):]
                continue
            frame_start = start + len(BOUNDARY)

            end = buf.find(JPEG_EOI, frame_start)
            if end == -1:
                continue
            return buf[frame_start:end + len(JPEG_EOI)]


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "snapshot.jpg"
    url = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_URL

    frame = fetch_one_frame(url)
    if not frame.startswith(JPEG_SOI):
        raise RuntimeError("Extracted data doesn't look like a JPEG (no SOI marker).")

    with open(out_path, "wb") as f:
        f.write(frame)
    print(f"Saved {out_path} ({len(frame)} bytes)")


if __name__ == "__main__":
    main()
