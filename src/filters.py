from __future__ import annotations

import cv2
import numpy as np

from .utils import ensure_odd, normalize_gray


def median_blur_masked(img: np.ndarray, ksize: int, mask: np.ndarray | None = None) -> np.ndarray:
    ksize = ensure_odd(ksize, 1)
    out = cv2.medianBlur(img, ksize)
    if mask is not None:
        out = out.copy()
        out[mask == 0] = img[mask == 0]
    return out


def gaussian_blur_masked(img: np.ndarray, ksize: int, sigma: float = 0.0, mask: np.ndarray | None = None) -> np.ndarray:
    ksize = ensure_odd(ksize, 1)
    out = cv2.GaussianBlur(img, (ksize, ksize), sigma)
    if mask is not None:
        out = out.copy()
        out[mask == 0] = img[mask == 0]
    return out


def maximum_filter_gray(img: np.ndarray, radius: int) -> np.ndarray:
    k = max(1, int(2 * radius + 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(img, kernel)


def minimum_filter_gray(img: np.ndarray, radius: int) -> np.ndarray:
    k = max(1, int(2 * radius + 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.erode(img, kernel)


def rolling_ball_background_subtraction(img: np.ndarray, radius: int) -> np.ndarray:
    radius = max(1, int(radius))
    background = gaussian_blur_masked(img, 2 * radius + 1, sigma=max(1.0, radius / 2.0))
    out = img.astype(np.int16) - background.astype(np.int16)
    return normalize_gray(out)


def subtract_ridge_suppressed_background(img: np.ndarray, radius: int) -> np.ndarray:
    radius = max(1, int(radius))
    bg = median_blur_masked(img, 2 * radius + 1)
    out = img.astype(np.int16) - bg.astype(np.int16)
    return normalize_gray(out)
