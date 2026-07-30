import os
import threading
import time

import cv2
import numpy as np
import psutil
from ai_edge_litert.interpreter import Interpreter
from fastapi.responses import StreamingResponse

from arduino.app_bricks.web_ui import WebUI
from arduino.app_peripherals.camera import Camera
from arduino.app_utils import App, Bridge, Logger
from arduino.app_utils.image import compress_to_jpeg

logger = Logger("PersonDetector")

logger.info("=== YOLO Person Detector starting up ===")

# --- 1. Load the model ---
logger.info("Loading YOLOv8n int8 tflite model from yolov8n_int8.tflite ...")
interpreter = Interpreter(model_path="yolov8n_int8.tflite", num_threads=4)
interpreter.allocate_tensors()
INPUT_DETAILS = interpreter.get_input_details()[0]
OUTPUT_DETAILS = interpreter.get_output_details()[0]
logger.info("Model loaded.")

INPUT_SIZE = 320  # fixed by export: model's input tensor shape is (1, 3, 320, 320)
CONF_THRESH = 0.4
NMS_THRESH = 0.45
PERSON_CLASS_ID = 0

logger.info("Initializing USB camera (640x480 @ 10fps)...")
camera = Camera(resolution=(640, 480), fps=10)

logger.info("Starting web UI server...")
ui = WebUI()

_frame_lock = threading.Lock()
_latest_jpeg = None
_person_present = False
_person_count = 0

_fps = 0.0
_last_frame_time = None

_process = psutil.Process(os.getpid())
_process.cpu_percent(interval=None)  # prime the internal baseline
psutil.cpu_percent(interval=None)  # prime the system-wide baseline


def preprocess(frame):
    img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    return np.expand_dims(img, axis=0)  # -> (1, 3, INPUT_SIZE, INPUT_SIZE)


def run_inference(input_tensor):
    interpreter.set_tensor(INPUT_DETAILS["index"], input_tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(OUTPUT_DETAILS["index"])
    return output[0]  # drop batch dim -> (84, num_anchors)


def postprocess(output, orig_w, orig_h):
    # YOLOv8 raw output shape: (84, num_anchors) -> transpose to (num_anchors, 84)
    # columns: [cx, cy, w, h, class0_score, class1_score, ...]
    output = output.T

    cls_scores = output[:, 4:]
    class_ids = np.argmax(cls_scores, axis=1)
    confidences = cls_scores[np.arange(len(cls_scores)), class_ids]

    mask = (class_ids == PERSON_CLASS_ID) & (confidences >= CONF_THRESH)
    if not np.any(mask):
        return [], []

    detections = output[mask]
    confidences = confidences[mask]

    cx, cy, w, h = detections[:, 0], detections[:, 1], detections[:, 2], detections[:, 3]
    # scale back from INPUT_SIZE to original frame size
    x_scale, y_scale = orig_w / INPUT_SIZE, orig_h / INPUT_SIZE
    x1 = (cx - w / 2) * x_scale
    y1 = (cy - h / 2) * y_scale
    bw = w * x_scale
    bh = h * y_scale

    boxes = np.stack([x1, y1, bw, bh], axis=1).tolist()
    scores = confidences.astype(float).tolist()

    # NMS
    indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESH, NMS_THRESH)
    if len(indices) == 0:
        return [], []
    indices = np.array(indices).flatten()
    return [boxes[i] for i in indices], [scores[i] for i in indices]


def notify_sketch(person_present):
    global _person_present
    if person_present == _person_present:
        return
    _person_present = person_present
    Bridge.notify("set_person_detected", person_present)


def get_status():
    return {
        "person_detected": _person_present,
        "count": _person_count,
        "fps": round(_fps, 1),
        "process_cpu_percent": _process.cpu_percent(interval=None),
        "process_memory_mb": round(_process.memory_info().rss / (1024 * 1024), 1),
        "system_cpu_percent": psutil.cpu_percent(interval=None),
        "system_memory_percent": psutil.virtual_memory().percent,
    }


def stream_video_feed():
    def generate():
        while True:
            with _frame_lock:
                jpeg_bytes = _latest_jpeg
            if jpeg_bytes is None:
                time.sleep(0.05)
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
            time.sleep(0.1)

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


ui.expose_api("GET", "/status", get_status)
ui.expose_api("GET", "/video_feed", stream_video_feed)


def loop():
    global _person_count, _fps, _last_frame_time

    frame = camera.capture()
    if frame is None:
        time.sleep(0.05)
        return

    now = time.monotonic()
    if _last_frame_time is not None:
        dt = now - _last_frame_time
        if dt > 0:
            instant_fps = 1.0 / dt
            _fps = instant_fps if _fps == 0.0 else (_fps * 0.9 + instant_fps * 0.1)
    _last_frame_time = now

    h, w = frame.shape[:2]
    mat_in = preprocess(frame)
    raw_output = run_inference(mat_in)
    boxes, scores = postprocess(raw_output, w, h)

    for (x, y, bw, bh), conf in zip(boxes, scores):
        cv2.rectangle(frame, (int(x), int(y)), (int(x + bw), int(y + bh)), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"person {conf:.2f}",
            (int(x), int(y) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    _person_count = len(boxes)
    notify_sketch(_person_count > 0)

    jpeg = compress_to_jpeg(frame, quality=70)
    if jpeg is not None:
        global _latest_jpeg
        with _frame_lock:
            _latest_jpeg = jpeg.tobytes()


camera.start()
logger.info("Camera started.")

logger.info(f"Web page available at: {ui.url}")
logger.info("Running person detection loop. Press Ctrl+C to stop.")

# See: https://docs.arduino.cc/software/app-lab/tutorials/getting-started/#app-run
App.run(user_loop=loop)
