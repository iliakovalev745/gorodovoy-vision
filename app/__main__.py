from datetime import datetime, timezone
import logging
import threading
import time

import cv2
from fastapi import FastAPI
import uvicorn

import config
from vision.detector import Detector
from vision.state import snapshot, status, update

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("gorodovoy-vision")
api = FastAPI(title="Gorodovoy Vision API", version="0.2.0")

stop = threading.Event()
worker = None


def source(s):
    return int(s) if str(s).strip().isdigit() else s


def run_vision():
    cap = None
    try:
        detector = Detector()
        src = source(config.VIDEO_SOURCE)
        cap = cv2.VideoCapture(src)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            raise RuntimeError(f"Не удалось открыть источник: {src!r}")

        status("running")
        log.info("Video source opened")

        t0 = time.perf_counter()
        count = 0
        idx = 0

        while not stop.is_set():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.2)
                continue

            idx += 1
            if (idx - 1) % config.FRAME_SKIP:
                continue

            start = time.perf_counter()
            objects = detector.predict(frame)
            ms = (time.perf_counter() - start) * 1000

            count += 1
            elapsed = time.perf_counter() - t0
            fps = count / elapsed if elapsed else 0
            h, w = frame.shape[:2]

            update(
                {
                    "status": "running",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "frame": {"width": w, "height": h},
                    "objects": objects,
                    "fps": round(fps, 2),
                    "inference_ms": round(ms, 2),
                }
            )

            if config.SHOW_WINDOW:
                view = frame.copy()
                for o in objects:
                    b = o["bbox"]
                    x1, y1, x2, y2 = map(
                        int, (b["x1"], b["y1"], b["x2"], b["y2"])
                    )
                    cv2.rectangle(view, (x1, y1), (x2, y2), (0, 220, 0), 2)

                    cx, cy = int(round(o["center"]["x"])), int(
                        round(o["center"]["y"])
                    )
                    cv2.circle(view, (cx, cy), 5, (0, 220, 255), -1)
                    cv2.drawMarker(
                        view, (cx, cy), (0, 220, 255), cv2.MARKER_CROSS, 14, 2
                    )

                    pos = o["relative_position"]
                    line1 = (
                        f"{o['label']} {o['confidence']:.2f} | "
                        f"{pos['bearing_deg']:+.1f} deg {pos['direction']}"
                    )
                    line2 = f"center: x={cx}, y={cy}"

                    text_y = max(18, y1 - 25)
                    for line_no, line in enumerate((line1, line2)):
                        yy = text_y + line_no * 19
                        (tw, th), _ = cv2.getTextSize(
                            line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                        )
                        yy = min(max(th + 4, yy), h - 4)
                        cv2.rectangle(
                            view,
                            (x1, yy - th - 4),
                            (min(w - 1, x1 + tw + 6), yy + 4),
                            (20, 35, 20),
                            -1,
                        )
                        cv2.putText(
                            view,
                            line,
                            (x1 + 3, yy),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 80),
                            2,
                            cv2.LINE_AA,
                        )

                cv2.putText(
                    view,
                    f"FPS {fps:.1f} | inference {ms:.1f} ms",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 220, 255),
                    2,
                )
                cv2.imshow("Gorodovoy Vision", view)
                if cv2.waitKey(1) & 255 in (ord("q"), 27):
                    stop.set()
                    break

    except Exception as e:
        log.exception("Vision worker failed")
        status("error", str(e))
    finally:
        if cap:
            cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


@api.on_event("startup")
def startup():
    global worker
    stop.clear()
    worker = threading.Thread(target=run_vision, daemon=True)
    worker.start()


@api.on_event("shutdown")
def shutdown():
    stop.set()
    if worker and worker.is_alive():
        worker.join(timeout=3)


@api.get("/health")
def health():
    d, m = snapshot()
    return {
        "ok": d["status"] == "running",
        "status": d["status"],
        "last_error": m["last_error"],
    }


@api.get("/api/v1/detections")
def detections():
    return snapshot()[0]


@api.get("/api/v1/metrics")
def metrics():
    return snapshot()[1]


if __name__ == "__main__":
    uvicorn.run(api, host=config.APP_HOST, port=config.APP_PORT)