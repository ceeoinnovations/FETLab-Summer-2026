# 😀 Homography test

Maps a USB webcam's pixel coordinates to floor coordinates (meters) around
the robot, using a one-time homography calibration. Runs entirely on the
MPU (Linux/Python) side — no MCU/sketch involvement is needed for this app.

## Coordinate system

Pin this down before measuring anything:

- **Origin**: robot base center, projected straight down to the floor.
- **X**: forward (direction robot faces). **Y**: to the robot's left. Units: meters.

Measure your 6 calibration points with a tape measure relative to that origin.

## Files

```
python/
├── main.py            # App Lab entry point — runs the continuous locate loop
├── calibrate.py        # one-off script: points.csv -> calibration.npz
├── locate.py            # detection + image->world mapping (used by main.py)
├── points.csv            # your 6 measured correspondences (fill in by hand)
└── requirements.txt        # opencv-python, numpy
```

## 1. Measure and fill in points.csv

Spread 6 points across the actual working area (near, far, left, right).
Avoid near-collinear points — 3+ points on a line degrades the fit. Four
points is the homography minimum; 6 gives an over-determined least-squares
fit and a reprojection error you can sanity-check.

To read off pixel coordinates, grab a still frame with the webcam pointed
at your marked floor points:

```bash
cd ~/ArduinoApps/homography-test/python
python3 snapshot.py
```

Pull `snapshot.jpg` off the board and open it in any image viewer/editor
that shows cursor pixel position, then hover over each of your 6 marked
floor points to read `(u, v)`.

Edit `python/points.csv`:

```csv
image_u,image_v,world_x,world_y
412,690,0.50,0.30
...  (6 rows total)
```

`image_u,image_v` are pixel coordinates read off a captured frame (e.g. open
a saved frame in an image viewer and hover to read pixel position).
`world_x,world_y` are the tape-measured floor coordinates in the frame above.

## 2. Calibrate

With the USB webcam connected, on the board:

```bash
cd ~/ArduinoApps/homography-test/python
python3 calibrate.py
```

This prints the reprojection RMSE — your go/no-go signal. If it's larger
than your tape-measure error (well over a few cm), you likely have a bad
correspondence or a mislabeled point; re-check `points.csv`. On success it
writes `calibration.npz` (`H`, `H_inv`, `image_size`, `units`, `rmse`) next
to it. Re-run this step any time the camera is moved, refocused, or
replaced.

## 3. Run the app

Via App Lab's Run button, or from the CLI:

```bash
arduino-app-cli app start ~/ArduinoApps/homography-test
arduino-app-cli app logs  ~/ArduinoApps/homography-test
```

`main.py` loads `calibration.npz`, opens the USB webcam, and repeatedly
detects the configured target and logs its floor position. Python logs go
to the app's log file, not stdout — use `arduino-app-cli app logs` to
follow them.

By default `main.py` uses the ArUco detector (`DETECTOR = "aruco"` in
`python/main.py`). To track a colored object instead, set `DETECTOR =
"color"` and fill in `COLOR_HSV_LOW` / `COLOR_HSV_HIGH`.

## Validation (do this — RMSE alone won't catch everything)

After calibrating, physically place the marker at a known floor coordinate
you did **not** use in calibration, run `python3 locate.py`, and compare.
This end-to-end test catches coordinate-frame sign errors (X/Y swapped or
flipped) that reprojection RMSE alone won't reveal.

`locate.draw_debug_overlay()` can draw the reprojected world grid onto a
frame via `H_inv` plus the detected point — useful for eyeballing whether
the floor mapping looks sane if you save a frame out for inspection.

## Notes

- `CAM_INDEX = 0` in both `calibrate.py` and `locate.py` assumes `/dev/video0`;
  change it if the robot has more than one camera.
- The webcam's capture resolution must stay the same between calibration and
  runtime — `locate.py` checks this and raises if it doesn't match.
