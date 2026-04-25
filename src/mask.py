from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from .utils import ensure_odd


def largest_component(mask01: np.ndarray, min_area: int = 1000) -> np.ndarray:
    mask01 = (mask01 > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask01, connectivity=8)
    if n <= 1:
        return mask01 * 255
    best_lab, best_area = 0, 0
    for lab in range(1, n):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area >= min_area and area > best_area:
            best_lab, best_area = lab, area
    out = np.zeros_like(mask01, dtype=np.uint8)
    if best_lab > 0:
        out[labels == best_lab] = 255
    return out


def fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    mask = ((mask > 0).astype(np.uint8) * 255)
    h, w = mask.shape[:2]
    flood = mask.copy()
    cv2.floodFill(flood, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 255)
    flood_inv = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask, flood_inv)


def clean_mask(mask: np.ndarray, open_k: int = 5, close_k: int = 9, min_area: int = 1000) -> np.ndarray:
    out = ((mask > 0).astype(np.uint8) * 255)
    if close_k > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k)
    if open_k > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k)
    out = fill_mask_holes(out)
    out = largest_component(out, min_area=min_area)
    return out


def mask_from_black_background(gray: np.ndarray, thresh: int = 35, invert: bool = False, open_k: int = 5,
                               close_k: int = 9, min_area: int = 1000) -> np.ndarray:
    if invert:
        mask = (gray <= thresh).astype(np.uint8) * 255
    else:
        mask = (gray > thresh).astype(np.uint8) * 255
    return clean_mask(mask, open_k=open_k, close_k=close_k, min_area=min_area)


def build_foreground_mask(gray: np.ndarray, blur_k: int = 9, block: int = 31, C: int = 7,
                          close_k: int = 9, open_k: int = 5, min_area: int = 4000) -> np.ndarray:
    blur_k = ensure_odd(blur_k, 1)
    block = ensure_odd(block, 3)
    sm = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
    diff = cv2.absdiff(gray, sm)
    diff = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(diff)
    mask = cv2.adaptiveThreshold(diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, int(C))
    return clean_mask(mask, open_k=open_k, close_k=close_k, min_area=min_area)


def crop_to_mask_bbox(img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        h, w = mask.shape[:2]
        return img.copy(), (0, 0, w, h)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return img[y0:y1, x0:x1].copy(), (x0, y0, x1 - x0, y1 - y0)


def apply_mask_outside_zero(img: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    if mask is None:
        return img.copy()
    out = img.copy()
    out[mask == 0] = 0
    return out


def apply_mask_outside_white(img: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    if mask is None:
        return img.copy()
    out = img.copy()
    out[mask == 0] = 255
    return out


def estimate_finger_width(mask: np.ndarray) -> float:
    pts = np.column_stack(np.where(mask > 0))
    if len(pts) < 5:
        return float(mask.shape[1])
    pts = pts[:, ::-1].astype(np.float32)
    (_, _), (w, h), _ = cv2.minAreaRect(pts)
    return float(max(min(w, h), 1.0))


def compute_finger_scale(mask: np.ndarray, ref_width: float = 500.0) -> float:
    width = estimate_finger_width(mask)
    return float(width / float(ref_width))

def compute_fingerprint_roi(
    gray: np.ndarray,
    block: int = 16,
    close_k: int = 25,
    erode_k: int = 7,
    close_iter: int = 4,
    dilate_iter: int = 1,
    min_area: int = 4000,
) -> np.ndarray:
    """
    Variance-based fingerprint ROI estimation.
    Regions with stronger local intensity variation are treated as foreground.
    """
    block = ensure_odd(block, 3)
    close_k = ensure_odd(close_k, 1)
    erode_k = ensure_odd(erode_k, 1)

    gray_f = gray.astype(np.float32)

    mean = cv2.blur(gray_f, (block, block))
    sqmean = cv2.blur(gray_f * gray_f, (block, block))
    var = sqmean - mean * mean

    var = cv2.normalize(var, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(var, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if close_k > 1:
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=close_iter)

    if erode_k > 1:
        k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_k, erode_k))
        mask = cv2.erode(mask, k_erode, iterations=1)
        if dilate_iter > 0:
            mask = cv2.dilate(mask, k_erode, iterations=dilate_iter)

    mask = fill_mask_holes(mask)
    mask = largest_component(mask, min_area=min_area)
    return mask

