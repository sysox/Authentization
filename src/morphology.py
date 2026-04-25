from __future__ import annotations

import cv2
import numpy as np
from .utils import ensure_odd


def morph_open(binary: np.ndarray, ksize: int, iterations: int = 1) -> np.ndarray:
    if ksize <= 1:
        return binary.copy()
    ksize = ensure_odd(ksize, 1)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, k, iterations=iterations)


def morph_close(binary: np.ndarray, ksize: int, iterations: int = 1) -> np.ndarray:
    if ksize <= 1:
        return binary.copy()
    ksize = ensure_odd(ksize, 1)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=iterations)


def morph_erode(binary: np.ndarray, ksize: int, iterations: int = 1) -> np.ndarray:
    if ksize <= 1:
        return binary.copy()
    ksize = ensure_odd(ksize, 1)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.erode(binary, k, iterations=iterations)


def morph_dilate(binary: np.ndarray, ksize: int, iterations: int = 1) -> np.ndarray:
    if ksize <= 1:
        return binary.copy()
    ksize = ensure_odd(ksize, 1)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.dilate(binary, k, iterations=iterations)


def thicken_ridges(binary: np.ndarray, ksize: int = 3, iterations: int = 1) -> np.ndarray:
    if ksize <= 1:
        return binary.copy()
    ksize = ensure_odd(ksize, 1)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.dilate(binary, k, iterations=iterations)


def remove_small_components(binary: np.ndarray, min_area: int = 10) -> np.ndarray:
    mask01 = (binary > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask01, connectivity=8)
    out = np.zeros_like(mask01, dtype=np.uint8)
    for lab in range(1, n):
        if int(stats[lab, cv2.CC_STAT_AREA]) >= int(min_area):
            out[labels == lab] = 255
    return out


def fill_holes(binary: np.ndarray) -> np.ndarray:
    binary = ((binary > 0).astype(np.uint8) * 255)
    h, w = binary.shape[:2]
    flood = binary.copy()
    cv2.floodFill(flood, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 255)
    flood_inv = cv2.bitwise_not(flood)
    return cv2.bitwise_or(binary, flood_inv)


def bridge_gaps(binary: np.ndarray, ksize: int = 3) -> np.ndarray:
    return morph_close(binary, ksize, iterations=1)


def despeckle_binary(binary: np.ndarray, min_area: int = 5) -> np.ndarray:
    return remove_small_components(binary, min_area=min_area)


def apply_morphology(
    binary: np.ndarray,
    close_k: int = 3,
    open_k: int = 2,
    min_area: int = 0,
    fill: bool = False,
    dilate_k: int = 0,
) -> np.ndarray:

    out = binary.copy()

    if close_k > 1:
        out = morph_close(out, close_k)

    if open_k > 1:
        out = morph_open(out, open_k)

    if fill:
        out = fill_holes(out)

    if min_area > 0:
        out = remove_small_components(out, min_area=min_area)

    if dilate_k > 1:
        out = thicken_ridges(out, dilate_k)

    return out
