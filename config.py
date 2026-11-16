
import json
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def boolean(name, default=False):
    return os.getenv(name, str(default)).strip().lower() in {
        "1", "true", "yes", "on"
    }


def project_path(name, default):
    value = Path(os.getenv(name, default))
    if not value.is_absolute():
        value = BASE_DIR / value
    return value


APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "8000"))

SOURCE_MODE = int(os.getenv("SOURCE_MODE", "0"))
VIDEO_SOURCE = os.getenv("VIDEO_SOURCE", "0")

RTSP_TRANSPORT = os.getenv("RTSP_TRANSPORT", "tcp")
RTSP_OPEN_TIMEOUT_MS = int(os.getenv("RTSP_OPEN_TIMEOUT_MS", "5000"))
RTSP_READ_TIMEOUT_MS = int(os.getenv("RTSP_READ_TIMEOUT_MS", "3000"))

INPUT_DIR = project_path("INPUT_DIR", "input_videos")
OUTPUT_DIR = project_path("OUTPUT_DIR", "output_videos")
RESULTS_DIR = project_path("RESULTS_DIR", "results")

CAMERA_OUTPUT_DIR = OUTPUT_DIR / "camera"
FILES_OUTPUT_DIR = OUTPUT_DIR / "files"

for directory in (
    INPUT_DIR,
    OUTPUT_DIR,
    RESULTS_DIR,
    CAMERA_OUTPUT_DIR,
    FILES_OUTPUT_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)

MODEL_PATH = os.getenv("MODEL_PATH", "yolo26n.pt")
if not Path(MODEL_PATH).is_absolute():
    MODEL_PATH = str(BASE_DIR / MODEL_PATH)

DEVICE = os.getenv("DEVICE", "auto")
IMGSZ = int(os.getenv("IMGSZ", "416"))
CONF = float(os.getenv("CONF", "0.35"))
FRAME_SKIP = max(1, int(os.getenv("FRAME_SKIP", "1")))
HALF_PRECISION = boolean("HALF_PRECISION", True)
MAX_DETECTIONS = int(os.getenv("MAX_DETECTIONS", "100"))

SHOW_WINDOW = boolean("SHOW_WINDOW", False)
SAVE_VIDEO = boolean("SAVE_VIDEO", True)
SAVE_METRICS = boolean("SAVE_METRICS", True)

JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "75"))
STREAM_INTERVAL = float(os.getenv("STREAM_INTERVAL", "0.03"))

CLASSES_FILE = os.getenv(
    "CLASSES_FILE",
    str(BASE_DIR / "classes.txt")
)

OBJECT_CLASSES = [
    item.strip().lower()
    for item in os.getenv("OBJECT_CLASSES", "").split(",")
    if item.strip()
]

CAMERA_HFOV_DEG = float(os.getenv("CAMERA_HFOV_DEG", "90"))
CAMERA_FX_PX = float(os.getenv("CAMERA_FX_PX", "0"))
ENABLE_DISTANCE_ESTIMATE = boolean(
    "ENABLE_DISTANCE_ESTIMATE",
    False
)

try:
    OBJECT_WIDTHS_M = json.loads(
        os.getenv(
            "OBJECT_WIDTHS_M",
            '{"person":0.45,"car":1.8,"bus":2.5,"truck":2.5}'
        )
    )
except json.JSONDecodeError:
    OBJECT_WIDTHS_M = {}
