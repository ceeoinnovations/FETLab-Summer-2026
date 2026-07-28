# ── Change these two values before running anything ───────────────────────────

# Bluetooth card serial number printed on the LEGO hub
SERIAL = 1227

# Smartphone camera stream URL.
# Android — install "IP Webcam" (free, Play Store):
#   start the server in the app, then use:  http://192.168.x.x:8080/video
# iOS — install "EpocCam" or "Camo":
#   follow the app's instructions for the stream URL
# Testing without a phone — use 0 for your laptop webcam
CAMERA = 1

# Minimum model confidence (0–1) required to act on a prediction.
# Below this threshold the motors stop. Raise it if commands are noisy,
# lower it if the robot is too hesitant.
CONFIDENCE_THRESHOLD = 0.75
