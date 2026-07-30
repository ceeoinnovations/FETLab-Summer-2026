import logging
import time
from pathlib import Path

from arduino.app_utils import App

import locate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

APP_DIR = Path(__file__).parent
CALIBRATION_PATH = APP_DIR / "calibration.npz"

DETECTOR = "aruco"  # or "color"
COLOR_HSV_LOW = (0, 0, 0)
COLOR_HSV_HIGH = (0, 0, 0)

_calib = None


def _load_calibration():
    if not CALIBRATION_PATH.exists():
        logger.warning(
            "No calibration.npz at %s - fill in points.csv and run "
            "'python3 calibrate.py' on the board first",
            CALIBRATION_PATH,
        )
        return None
    calib = locate.load_calibration(str(CALIBRATION_PATH))
    logger.info("Loaded calibration, rmse=%.4f m", calib["rmse"])
    return calib


def loop():
    """Called repeatedly by the App framework."""
    global _calib
    if _calib is None:
        _calib = _load_calibration()
        if _calib is None:
            time.sleep(5)
            return

    try:
        result = locate.locate_once(
            _calib, detector=DETECTOR, hsv_low=COLOR_HSV_LOW, hsv_high=COLOR_HSV_HIGH
        )
    except RuntimeError as e:
        logger.error("Locate failed: %s", e)
        time.sleep(1)
        return

    if result is None:
        logger.info("No object detected")
    else:
        x, y = result["x"], result["y"]
        logger.info("Object at X=%.2f Y=%.2f m", x, y)

    time.sleep(0.2)


# See: https://docs.arduino.cc/software/app-lab/tutorials/getting-started/#app-run
App.run(user_loop=loop)
