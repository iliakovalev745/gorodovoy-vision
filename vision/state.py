
import threading
import time

lock = threading.Lock()

latest = {
    "status": "starting",
    "timestamp": None,
    "frame": None,
    "objects": [],
    "fps": 0.0,
    "camera_fps": 0.0,
    "capture_fps": 0.0,
    "inference_ms": 0.0,
    "processing_ms": 0.0,
}

metrics = {
    "frames_processed": 0,
    "camera_fps": 0.0,
    "capture_fps": 0.0,
    "processing_fps": 0.0,
    "inference_ms": 0.0,
    "processing_ms": 0.0,
    "last_error": None,
    "started_at": time.time(),
}

latest_jpeg = None


def update(data):
    with lock:
        latest.update(data)

        metrics["frames_processed"] += 1
        metrics["camera_fps"] = data.get(
            "camera_fps",
            metrics["camera_fps"]
        )
        metrics["capture_fps"] = data.get(
            "capture_fps",
            metrics["capture_fps"]
        )
        metrics["processing_fps"] = data.get(
            "fps",
            metrics["processing_fps"]
        )
        metrics["inference_ms"] = data.get(
            "inference_ms",
            metrics["inference_ms"]
        )
        metrics["processing_ms"] = data.get(
            "processing_ms",
            metrics["processing_ms"]
        )


def update_image(image):
    global latest_jpeg

    with lock:
        latest_jpeg = image


def get_image():
    with lock:
        return latest_jpeg


def status(value, error=None):
    with lock:
        latest["status"] = value
        metrics["last_error"] = error


def snapshot():
    with lock:
        return dict(latest), dict(metrics)
