
import csv
import json
from pathlib import Path

import cv2


class Recorder:
    def __init__(self, directory, name, fps, width, height):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

        self.video_path = self.directory / f"{name}.mp4"
        self.csv_path = self.directory / "metrics.csv"
        self.summary_path = self.directory / "summary.json"

        self.writer = cv2.VideoWriter(
            str(self.video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            max(1.0, float(fps)),
            (int(width), int(height))
        )

        if not self.writer.isOpened():
            raise RuntimeError(
                f"Не удалось создать видео: {self.video_path}"
            )

        self.csv_file = self.csv_path.open(
            "w",
            newline="",
            encoding="utf-8-sig"
        )

        self.csv_writer = csv.DictWriter(
            self.csv_file,
            fieldnames=[
                "timestamp",
                "frame",
                "camera_fps",
                "capture_fps",
                "processing_fps",
                "inference_ms",
                "processing_ms",
                "detections",
                "objects"
            ]
        )

        self.csv_writer.writeheader()

        self.frames = 0
        self.inference_total = 0.0
        self.processing_total = 0.0
        self.detections_total = 0

    def write(self, frame, data):
        self.writer.write(frame)

        objects = data.get("objects", [])

        self.csv_writer.writerow({
            "timestamp": data.get("timestamp"),
            "frame": data.get("frame", {}).get("index"),
            "camera_fps": data.get("camera_fps"),
            "capture_fps": data.get("capture_fps"),
            "processing_fps": data.get("fps"),
            "inference_ms": data.get("inference_ms"),
            "processing_ms": data.get("processing_ms"),
            "detections": len(objects),
            "objects": json.dumps(
                objects,
                ensure_ascii=False
            )
        })

        self.frames += 1
        self.inference_total += data.get("inference_ms", 0)
        self.processing_total += data.get("processing_ms", 0)
        self.detections_total += len(objects)

    def close(self):
        self.writer.release()
        self.csv_file.close()

        summary = {
            "frames_processed": self.frames,
            "average_inference_ms": (
                round(self.inference_total / self.frames, 2)
                if self.frames else 0
            ),
            "average_processing_ms": (
                round(self.processing_total / self.frames, 2)
                if self.frames else 0
            ),
            "total_detections": self.detections_total,
            "video": str(self.video_path),
            "metrics": str(self.csv_path)
        }

        self.summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=4),
            encoding="utf-8"
        )
