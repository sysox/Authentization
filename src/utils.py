from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


def ensure_odd(k: int, minimum: int = 1) -> int:
    k = max(minimum, int(k))
    if k % 2 == 0:
        k += 1
    return k


def normalize_gray(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    mn, mx = float(np.min(img)), float(np.max(img))
    if mx <= mn:
        return np.zeros_like(img, dtype=np.uint8)
    out = 255.0 * (img - mn) / (mx - mn)
    return np.clip(out, 0, 255).astype(np.uint8)


def resize_keep_aspect(img: np.ndarray, width: Optional[int]) -> np.ndarray:
    if width is None:
        return img
    h, w = img.shape[:2]
    if w == width:
        return img
    scale = width / float(w)
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(img, (width, new_h), interpolation=interp)


def as_uint8_binary(img: np.ndarray) -> np.ndarray:
    return ((img > 0).astype(np.uint8) * 255)


def bool01(img: np.ndarray) -> np.ndarray:
    return (img > 0).astype(np.uint8)


def safe_div(num: float, den: float, default: float = 0.0) -> float:
    return default if den == 0 else float(num) / float(den)


def pairwise_distances(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.zeros((0, 0), dtype=np.float32)
    p = points.astype(np.float32)
    diff = p[:, None, :] - p[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def crop_from_bbox(img: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = bbox
    return img[y:y + h, x:x + w].copy()
