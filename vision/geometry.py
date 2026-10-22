import math
import config


def focal_px(width):
    if config.CAMERA_FX_PX > 0:
        return config.CAMERA_FX_PX
    return width / (
        2
        * math.tan(
            math.radians(max(1, min(179, config.CAMERA_HFOV_DEG))) / 2
        )
    )


def relative_position(label, bbox, w, h):
    x1, y1, x2, y2 = [float(bbox[k]) for k in ("x1", "y1", "x2", "y2")]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    fx = focal_px(w)
    angle = math.degrees(math.atan((cx - w / 2) / fx))
    direction = (
        "прямо"
        if abs(angle) < 1
        else ("слева" if angle < 0 else "справа")
    )
    distance = None
    method = "не рассчитывалось"
    rw = config.OBJECT_WIDTHS_M.get(label)
    pw = x2 - x1
    if config.ENABLE_DISTANCE_ESTIMATE and rw and pw > 1:
        distance = round(rw * fx / pw, 2)
        method = "оценка по ширине объекта; требует калибровки"
    return {
        "image_offset": {
            "x_normalized": round((cx - w / 2) / (w / 2), 5),
            "y_normalized": round((cy - h / 2) / (h / 2), 5),
        },
        "bearing_deg": round(angle, 2),
        "direction": direction,
        "distance_m": distance,
        "distance_method": method,
    }