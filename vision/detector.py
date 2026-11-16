
import logging
from pathlib import Path

import torch
from ultralytics import YOLO

import config
from vision.geometry import relative_position

log = logging.getLogger(__name__)


def wanted_classes():
    if config.OBJECT_CLASSES:
        return set(config.OBJECT_CLASSES)

    path = Path(config.CLASSES_FILE)

    if path.exists():
        return {
            line.strip().lower()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }

    return set()


class Detector:
    def __init__(self):
        log.info("Loading model: %s", config.MODEL_PATH)

        self.model = YOLO(config.MODEL_PATH)

        if config.DEVICE.lower() == "auto":
            self.device = "0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = config.DEVICE

        if self.device != "cpu" and torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        log.info("Device: %s", self.device)
        log.info("Image size: %s", config.IMGSZ)

        wanted = wanted_classes()
        self.ids = None

        if wanted:
            self.ids = [
                int(class_id)
                for class_id, name in self.model.names.items()
                if str(name).lower() in wanted
            ]

            found = {
                str(self.model.names[class_id]).lower()
                for class_id in self.ids
            }

            missing = wanted - found

            if missing:
                log.warning(
                    "Unsupported classes: %s",
                    ", ".join(sorted(missing))
                )

            if not self.ids:
                log.warning("Class filter disabled")

    def predict(self, frame):
        height, width = frame.shape[:2]

        kwargs = {
            "imgsz": config.IMGSZ,
            "conf": config.CONF,
            "verbose": False,
            "max_det": config.MAX_DETECTIONS,
            "device": self.device
        }

        if self.ids:
            kwargs["classes"] = self.ids

        with torch.inference_mode():
            result = self.model.predict(frame, **kwargs)[0]

        objects = []

        if result.boxes is None:
            return objects

        for box in result.boxes:
            class_id = int(box.cls[0].item())
            label = str(result.names.get(class_id, class_id))

            x1, y1, x2, y2 = map(
                float,
                box.xyxy[0].tolist()
            )

            bbox = {
                key: round(value, 2)
                for key, value in zip(
                    ("x1", "y1", "x2", "y2"),
                    (x1, y1, x2, y2)
                )
            }

            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            objects.append({
                "label": label,
                "confidence": round(float(box.conf[0].item()), 5),
                "bbox": bbox,
                "center": {
                    "x": round(center_x, 2),
                    "y": round(center_y, 2)
                },
                "relative": {
                    "center_x": round(center_x / width, 6),
                    "center_y": round(center_y / height, 6),
                    "width": round((x2 - x1) / width, 6),
                    "height": round((y2 - y1) / height, 6)
                },
                "relative_position": relative_position(
                    label,
                    bbox,
                    width,
                    height
                )
            })

        return objects
