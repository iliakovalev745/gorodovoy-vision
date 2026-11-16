
import os
import logging
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime

os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    f"rtsp_transport;{os.getenv('RTSP_TRANSPORT', 'tcp')}|fflags;nobuffer|max_delay;500000"
)

import cv2
import uvicorn

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse

import config

from vision.detector import Detector
from vision.state import (
    snapshot,
    status,
    update,
    update_image,
    get_image
)
from vision.recorder import Recorder


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)

log = logging.getLogger("gorodovoy-vision")

cv2.setNumThreads(1)

stop = threading.Event()
worker = None
preview_window_created = False

PREVIEW_MAX_WIDTH = int(os.getenv("PREVIEW_MAX_WIDTH", "1280"))
PREVIEW_MAX_HEIGHT = int(os.getenv("PREVIEW_MAX_HEIGHT", "720"))
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "80"))


def timestamp():
    return datetime.now().astimezone().isoformat(
        timespec="milliseconds"
    )


def source_value(value):
    value = str(value).strip()
    return int(value) if value.isdigit() else value


def create_session():
    name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    directory = config.CAMERA_OUTPUT_DIR / name
    directory.mkdir(parents=True, exist_ok=True)
    return name, directory


def draw_text(frame, text, position, color, scale, thickness):
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )


def annotate(frame, objects, data):
    view = frame.copy()
    height, width = view.shape[:2]

    preview_scale = min(
        PREVIEW_MAX_WIDTH / width,
        PREVIEW_MAX_HEIGHT / height,
        1.0
    )

    if preview_scale <= 0:
        preview_scale = 1.0

    stats_scale = 0.78 / preview_scale
    label_scale = 0.68 / preview_scale

    stats_thickness = max(2, round(2 / preview_scale))
    label_thickness = max(2, round(2 / preview_scale))
    box_thickness = max(2, round(2 / preview_scale))
    point_radius = max(4, round(5 / preview_scale))

    yellow = (0, 220, 255)
    green = (0, 220, 0)

    for obj in objects:
        bbox = obj["bbox"]

        x1, y1, x2, y2 = map(
            int,
            (
                bbox["x1"],
                bbox["y1"],
                bbox["x2"],
                bbox["y2"]
            )
        )

        cv2.rectangle(
            view,
            (x1, y1),
            (x2, y2),
            green,
            box_thickness
        )

        cx = int(obj["center"]["x"])
        cy = int(obj["center"]["y"])

        cv2.circle(
            view,
            (cx, cy),
            point_radius,
            yellow,
            -1
        )

        position = obj["relative_position"]

        line1 = (
            f'{obj["label"]} '
            f'{obj["confidence"]:.2f} '
            f'angle={position["bearing_deg"]:+.1f}'
        )

        line2 = (
            f'X={cx} Y={cy} '
            f'{position["direction"]}'
        )

        text_x = max(5, x1)
        text_offset = round(30 / preview_scale)
        text_gap = round(26 / preview_scale)

        text_y = max(
            round(28 / preview_scale),
            y1 - text_offset
        )

        draw_text(
            view,
            line1,
            (text_x, text_y),
            yellow,
            label_scale,
            label_thickness
        )

        draw_text(
            view,
            line2,
            (text_x, text_y + text_gap),
            yellow,
            label_scale,
            label_thickness
        )

    overlay = [
        f'Date: {data["timestamp"]}',
        f'Camera FPS: {data["camera_fps"]:.1f}',
        f'Capture FPS: {data["capture_fps"]:.1f}',
        f'Processing FPS: {data["fps"]:.2f}',
        f'Inference: {data["inference_ms"]:.1f} ms',
        f'Processing: {data["processing_ms"]:.1f} ms',
        f'Objects: {len(objects)}',
        f'Resolution: {width}x{height}'
    ]

    line_height = round(34 / preview_scale)
    start_y = round(38 / preview_scale)
    left_margin = round(16 / preview_scale)

    for index, line in enumerate(overlay):
        y = start_y + index * line_height

        if y >= height - 5:
            break

        draw_text(
            view,
            line,
            (left_margin, y),
            yellow,
            stats_scale,
            stats_thickness
        )

    return view


def show_preview(frame):
    global preview_window_created

    height, width = frame.shape[:2]

    scale = min(
        PREVIEW_MAX_WIDTH / width,
        PREVIEW_MAX_HEIGHT / height,
        1.0
    )

    preview_width = max(1, int(width * scale))
    preview_height = max(1, int(height * scale))

    if scale < 1.0:
        preview = cv2.resize(
            frame,
            (preview_width, preview_height),
            interpolation=cv2.INTER_AREA
        )
    else:
        preview = frame

    if not preview_window_created:
        cv2.namedWindow(
            "Gorodovoy Vision",
            cv2.WINDOW_NORMAL
        )
        cv2.resizeWindow(
            "Gorodovoy Vision",
            preview_width,
            preview_height
        )
        preview_window_created = True

    cv2.imshow("Gorodovoy Vision", preview)

    key = cv2.waitKey(1) & 0xFF

    if key in (ord("q"), 27):
        stop.set()
        return False

    return True


def close_preview():
    global preview_window_created

    if config.SHOW_WINDOW:
        cv2.destroyAllWindows()

    preview_window_created = False


def publish_frame(view):
    encoded, buffer = cv2.imencode(
        ".jpg",
        view,
        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    )

    if encoded:
        update_image(buffer.tobytes())


class LatestFrame:
    def __init__(self, source):
        self.source = source
        self.capture = None
        self.frame = None
        self.frame_id = 0
        self.lock = threading.Lock()
        self.thread = None
        self.running = False
        self.read_fps = 0.0

    def open_capture(self):
        if isinstance(self.source, str) and self.source.startswith("rtsp://"):
            capture = cv2.VideoCapture()
            params = []

            if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
                params.extend([
                    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                    int(getattr(config, "RTSP_OPEN_TIMEOUT_MS", 5000))
                ])

            if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
                params.extend([
                    cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                    int(getattr(config, "RTSP_READ_TIMEOUT_MS", 5000))
                ])

            try:
                opened = capture.open(
                    self.source,
                    cv2.CAP_FFMPEG,
                    params
                )
            except (TypeError, cv2.error):
                opened = capture.open(
                    self.source,
                    cv2.CAP_FFMPEG
                )

            if not opened:
                capture.release()
                raise RuntimeError("Не удалось открыть RTSP-поток")

            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return capture

        capture = cv2.VideoCapture(self.source)

        if not capture.isOpened():
            capture.release()
            raise RuntimeError(
                f"Не удалось открыть источник: {self.source}"
            )

        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture

    def start(self):
        self.capture = self.open_capture()
        self.running = True

        self.thread = threading.Thread(
            target=self.read_loop,
            daemon=True
        )
        self.thread.start()

    def read_loop(self):
        count = 0
        started = time.perf_counter()

        while self.running and not stop.is_set():
            ok, frame = self.capture.read()

            if not ok:
                time.sleep(0.01)
                continue

            with self.lock:
                self.frame = frame
                self.frame_id += 1

            count += 1
            elapsed = time.perf_counter() - started

            if elapsed >= 1.0:
                self.read_fps = count / elapsed
                count = 0
                started = time.perf_counter()

    def get(self):
        with self.lock:
            if self.frame is None:
                return None, self.frame_id

            return self.frame, self.frame_id

    def close(self):
        self.running = False

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)

        if self.capture is not None:
            self.capture.release()


def process_camera(detector):
    source = source_value(config.VIDEO_SOURCE)
    name, directory = create_session()

    log.info("Camera session: %s", directory)

    stream = LatestFrame(source)
    recorder = None

    try:
        stream.start()

        width = int(stream.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(stream.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        nominal_fps = stream.capture.get(cv2.CAP_PROP_FPS)

        if nominal_fps <= 0 or nominal_fps > 120:
            nominal_fps = 20.0

        if config.SAVE_VIDEO:
            recorder = Recorder(
                directory,
                "detection",
                nominal_fps,
                width,
                height
            )

        log.info(
            "Camera opened: %sx%s, nominal FPS: %.2f",
            width,
            height,
            nominal_fps
        )

        processed = 0
        started = time.perf_counter()
        last_frame_id = 0

        while not stop.is_set():
            frame, frame_id = stream.get()

            if frame is None or frame_id == last_frame_id:
                time.sleep(0.002)
                continue

            last_frame_id = frame_id
            process_start = time.perf_counter()

            inference_start = time.perf_counter()
            objects = detector.predict(frame)
            inference_ms = (
                time.perf_counter() - inference_start
            ) * 1000

            processed += 1

            elapsed = time.perf_counter() - started
            processing_fps = processed / elapsed if elapsed > 0 else 0

            processing_ms = (
                time.perf_counter() - process_start
            ) * 1000

            data = {
                "status": "running",
                "timestamp": timestamp(),
                "frame": {
                    "width": frame.shape[1],
                    "height": frame.shape[0],
                    "index": frame_id
                },
                "objects": objects,
                "camera_fps": round(nominal_fps, 2),
                "capture_fps": round(stream.read_fps, 2),
                "fps": round(processing_fps, 2),
                "inference_ms": round(inference_ms, 2),
                "processing_ms": round(processing_ms, 2)
            }

            view = annotate(frame, objects, data)
            update(data)

            if recorder is not None:
                recorder.write(view, data)

            publish_frame(view)

            if config.SHOW_WINDOW:
                if not show_preview(view):
                    break

    finally:
        stream.close()

        if recorder is not None:
            recorder.close()

        close_preview()


def process_file(path, detector):
    capture = cv2.VideoCapture(str(path))

    if not capture.isOpened():
        raise RuntimeError(
            f"Не удалось открыть видеофайл: {path}"
        )

    fps = capture.get(cv2.CAP_PROP_FPS)

    if fps <= 0 or fps > 120:
        fps = 25.0

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_skip = max(1, int(config.FRAME_SKIP))
    output_fps = max(1.0, fps / frame_skip)

    output_directory = (
        config.FILES_OUTPUT_DIR
        / (
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            + "_"
            + path.stem
        )
    )

    output_directory.mkdir(parents=True, exist_ok=True)

    recorder = None

    if config.SAVE_VIDEO:
        recorder = Recorder(
            output_directory,
            path.stem,
            output_fps,
            width,
            height
        )

    log.info(
        "Processing file: %s (%sx%s, %.2f FPS)",
        path.name,
        width,
        height,
        fps
    )

    frame_index = 0
    processed = 0
    started = time.perf_counter()

    try:
        while not stop.is_set():
            ok, frame = capture.read()

            if not ok:
                break

            frame_index += 1

            if (frame_index - 1) % frame_skip != 0:
                continue

            process_start = time.perf_counter()

            inference_start = time.perf_counter()
            objects = detector.predict(frame)
            inference_ms = (
                time.perf_counter() - inference_start
            ) * 1000

            processed += 1

            elapsed = time.perf_counter() - started
            processing_fps = processed / elapsed if elapsed > 0 else 0

            processing_ms = (
                time.perf_counter() - process_start
            ) * 1000

            data = {
                "status": "running",
                "timestamp": timestamp(),
                "frame": {
                    "width": frame.shape[1],
                    "height": frame.shape[0],
                    "index": frame_index
                },
                "objects": objects,
                "camera_fps": round(fps, 2),
                "capture_fps": 0.0,
                "fps": round(processing_fps, 2),
                "inference_ms": round(inference_ms, 2),
                "processing_ms": round(processing_ms, 2)
            }

            view = annotate(frame, objects, data)
            update(data)

            if recorder is not None:
                recorder.write(view, data)

            publish_frame(view)

            if config.SHOW_WINDOW:
                if not show_preview(view):
                    break

    finally:
        capture.release()

        if recorder is not None:
            recorder.close()

        close_preview()

    log.info(
        "Finished file: %s, processed frames: %s",
        path.name,
        processed
    )


def run_files(detector):
    extensions = {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".m4v"
    }

    files = sorted(
        path for path in config.INPUT_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    )

    if not files:
        raise RuntimeError(
            f"В папке {config.INPUT_DIR} нет видеофайлов"
        )

    for path in files:
        if stop.is_set():
            break

        process_file(path, detector)


def run_vision():
    try:
        status("starting")

        detector = Detector()

        if config.SOURCE_MODE == 0:
            status("running")
            process_camera(detector)

        elif config.SOURCE_MODE == 1:
            status("running")
            run_files(detector)

        else:
            raise ValueError(
                "SOURCE_MODE должен быть 0 или 1"
            )

        if stop.is_set():
            status("stopped")
        else:
            status("completed")

    except Exception as error:
        log.exception("Vision worker failed")
        status("error", str(error))


@asynccontextmanager
async def lifespan(app):
    global worker

    stop.clear()

    worker = threading.Thread(
        target=run_vision,
        daemon=True
    )
    worker.start()

    yield

    stop.set()

    if worker and worker.is_alive():
        worker.join(timeout=5)

    close_preview()


api = FastAPI(
    title="Gorodovoy Vision API",
    version="0.6.0",
    lifespan=lifespan
)


@api.get("/", response_class=HTMLResponse)
def dashboard():
    return """
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Городовой — Vision</title>
        <style>
            body {
                margin: 0;
                padding: 24px;
                background: #10151d;
                color: white;
                font-family: Arial, sans-serif;
            }
            h1 { margin-top: 0; }
            .layout {
                display: grid;
                grid-template-columns: 2fr 1fr;
                gap: 20px;
            }
            .panel {
                background: #1c2633;
                padding: 16px;
                border-radius: 12px;
                min-width: 0;
            }
            img {
                width: 100%;
                border-radius: 8px;
                display: block;
            }
            .metric {
                padding: 12px;
                margin: 8px 0;
                background: #283647;
                border-radius: 8px;
            }
            .objects {
                max-height: 350px;
                overflow-y: auto;
            }
            @media(max-width: 800px) {
                .layout { grid-template-columns: 1fr; }
            }
        </style>
    </head>
    <body>
        <h1>Городовой — система технического зрения</h1>
        <div class="layout">
            <div class="panel">
                <h3>Видеопоток</h3>
                <img src="/video_feed">
            </div>
            <div class="panel">
                <h3>Производительность</h3>
                <div id="metrics"></div>
                <h3>Обнаруженные объекты</h3>
                <div id="objects" class="objects"></div>
            </div>
        </div>
        <script>
            async function refresh() {
                try {
                    const [m, d] = await Promise.all([
                        fetch('/api/v1/metrics').then(r => r.json()),
                        fetch('/api/v1/detections').then(r => r.json())
                    ]);

                    document.getElementById('metrics').innerHTML = `
                        <div class="metric">Статус: ${d.status}</div>
                        <div class="metric">FPS камеры: ${m.camera_fps ?? m.fps}</div>
                        <div class="metric">FPS обработки: ${m.processing_fps ?? m.fps}</div>
                        <div class="metric">FPS чтения: ${m.capture_fps ?? 0}</div>
                        <div class="metric">Инференс: ${m.inference_ms} ms</div>
                        <div class="metric">Полная обработка: ${m.processing_ms ?? 0} ms</div>
                        <div class="metric">Обработано кадров: ${m.frames_processed}</div>
                        <div class="metric">Ошибка: ${m.last_error || 'Нет'}</div>
                    `;

                    document.getElementById('objects').innerHTML =
                        d.objects.map(o => `
                            <div class="metric">
                                <b>${o.label}</b><br>
                                Уверенность: ${o.confidence}<br>
                                X: ${o.center.x}, Y: ${o.center.y}<br>
                                Угол: ${o.relative_position.bearing_deg}°
                            </div>
                        `).join('') || 'Объекты не обнаружены';

                } catch (e) {
                    console.error(e);
                }
            }

            refresh();
            setInterval(refresh, 1000);
        </script>
    </body>
    </html>
    """


@api.get("/video_feed")
def video_feed():
    def generate():
        last_image = None

        while True:
            frame = get_image()

            if frame is not None and frame is not last_image:
                last_image = frame

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + frame
                    + b"\r\n"
                )

            time.sleep(float(getattr(config, "STREAM_INTERVAL", 0.05)))

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@api.get("/health")
def health():
    data, metrics = snapshot()

    return {
        "ok": data["status"] == "running",
        "status": data["status"],
        "last_error": metrics["last_error"]
    }


@api.get("/api/v1/detections")
def detections():
    return snapshot()[0]


@api.get("/api/v1/metrics")
def get_metrics():
    return snapshot()[1]


if __name__ == "__main__":
    uvicorn.run(
        api,
        host=config.APP_HOST,
        port=config.APP_PORT
    )
