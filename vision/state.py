import threading
import time

lock = threading.Lock()
latest = {
    "status": "starting",
    "timestamp": None,
    "frame": None,
    "objects": [],
    "fps": 0.0,
    "inference_ms": 0.0,
}
metrics = {
    "frames_processed": 0,
    "fps": 0.0,
    "inference_ms": 0.0,
    "last_error": None,
    "started_at": time.time(),
}


def update(p):
    with lock:
        latest.update(p)
        metrics["frames_processed"] += 1
        metrics["fps"] = p.get("fps", metrics["fps"])
        metrics["inference_ms"] = p.get("inference_ms", metrics["inference_ms"])


def status(s, e=None):
    with lock:
        latest["status"] = s
        metrics["last_error"] = e


def snapshot():
    with lock:
        return dict(latest), dict(metrics)