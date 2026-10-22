import os
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

# .env читается из папки, откуда пользователь запускает программу.
load_dotenv()


def resource_path(relative_path: str) -> str:
    """Путь к ресурсу в обычном проекте или внутри PyInstaller."""
    if getattr(sys, "frozen", False):
        base_path = Path(sys._MEIPASS)
    else:
        base_path = Path(__file__).resolve().parent

    return str(base_path / relative_path)


def boolean(name, default=False):
    return os.getenv(name, str(default)).lower() in {
        "1", "true", "yes", "on"
    }


APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "8000"))
VIDEO_SOURCE = os.getenv("VIDEO_SOURCE", "0")

bundled_model = Path(resource_path("models/yolo26n.pt"))
MODEL_PATH = os.getenv(
    "MODEL_PATH",
    str(bundled_model) if bundled_model.exists() else "yolo26n.pt",
)

DEVICE = os.getenv("DEVICE", "auto")
IMGSZ = int(os.getenv("IMGSZ", "640"))
CONF = float(os.getenv("CONF", "0.35"))
FRAME_SKIP = max(1, int(os.getenv("FRAME_SKIP", "1")))
SHOW_WINDOW = boolean("SHOW_WINDOW", True)

CLASSES_FILE = os.getenv(
    "CLASSES_FILE",
    resource_path("classes.txt"),
)

OBJECT_CLASSES = [
    x.strip().lower()
    for x in os.getenv("OBJECT_CLASSES", "").split(",")
    if x.strip()
]

CAMERA_HFOV_DEG = float(os.getenv("CAMERA_HFOV_DEG", "90"))
CAMERA_FX_PX = float(os.getenv("CAMERA_FX_PX", "0"))
ENABLE_DISTANCE_ESTIMATE = boolean("ENABLE_DISTANCE_ESTIMATE", False)

try:
    OBJECT_WIDTHS_M = json.loads(
        os.getenv(
            "OBJECT_WIDTHS_M",
            '{"person":0.45,"car":1.8,"bus":2.5,"truck":2.5,'
            '"motorcycle":0.8,"bicycle":0.6,"dog":0.5,"cat":0.25,'
            '"horse":0.7,"cow":0.8,"bear":0.8}',
        )
    )
except json.JSONDecodeError:
    OBJECT_WIDTHS_M = {}