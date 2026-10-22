from pathlib import Path
import logging

from ultralytics import YOLO

import config
from vision.geometry import relative_position

log = logging.getLogger(__name__)


def wanted_classes():
    if config.OBJECT_CLASSES:
        return set(config.OBJECT_CLASSES)
    p = Path(config.CLASSES_FILE)
    if p.exists():
        return {
            s.strip().lower()
            for s in p.read_text(encoding="utf-8").splitlines()
            if s.strip() and not s.lstrip().startswith("#")
        }
    return set()


class Detector:

    def __init__(self):
        log.info("Loading object model %s", config.MODEL_PATH)
        self.model = YOLO(config.MODEL_PATH)
        wanted = wanted_classes()
        self.ids = None
        if wanted:
            self.ids = [
                int(i)
                for i, n in self.model.names.items()
                if str(n).lower() in wanted
            ]
            missing = wanted - {
                str(self.model.names[i]).lower() for i in self.ids
            }
            if missing:
                log.warning(
                    "Classes not supported by model: %s",
                    ", ".join(sorted(missing)),
                )
            if not self.ids:
                log.warning("No matching class found; filter disabled")

    def predict(self, frame):
        h, w = frame.shape[:2]
        kw = {
            "imgsz": config.IMGSZ,
            "conf": config.CONF,
            "verbose": False,
        }
        if self.ids:
            kw["classes"] = self.ids
        if config.DEVICE.lower() not in {"", "auto"}:
            kw["device"] = config.DEVICE

        r = self.model.predict(frame, **kw)[0]
        out = []
        if r.boxes is not None:
            for b in r.boxes:
                i = int(b.cls[0].item())
                label = str(r.names.get(i, i))
                x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
                box = {
                    k: round(v, 2)
                    for k, v in zip(("x1", "y1", "x2", "y2"), (x1, y1, x2, y2))
                }
                out.append(
                    {
                        "label": label,
                        "confidence": round(float(b.conf[0].item()), 5),
                        "bbox": box,
                        "center": {
                            "x": round((x1 + x2) / 2, 2),
                            "y": round((y1 + y2) / 2, 2),
                        },
                        "relative": {
                            "center_x": round((x1 + x2) / 2 / w, 6),
                            "center_y": round((y1 + y2) / 2 / h, 6),
                            "width": round((x2 - x1) / w, 6),
                            "height": round((y2 - y1) / h, 6),
                        },
                        "relative_position": relative_position(
                            label, box, w, h
                        ),
                    }
                )
        return out