from __future__ import annotations

import cv2
import numpy as np
from .enhancement import median_blur
from .utils import ensure_odd

try:
    from skimage.filters import threshold_li as sk_threshold_li
    from skimage.filters import threshold_sauvola, threshold_niblack, threshold_triangle
    SKIMAGE_THRESH = True
except Exception:
    SKIMAGE_THRESH = False


def fixed_binarize(img_gray: np.ndarray, thresh: int = 127) -> np.ndarray:
    _, binary = cv2.threshold(img_gray, int(thresh), 255, cv2.THRESH_BINARY)
    return binary


def otsu_binarize(img_gray: np.ndarray) -> np.ndarray:
    _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def li_binarize(img_gray: np.ndarray) -> np.ndarray:
    if SKIMAGE_THRESH:
        t = float(sk_threshold_li(img_gray))
        return ((img_gray > t).astype(np.uint8) * 255)
    return otsu_binarize(img_gray)


def triangle_binarize(img_gray: np.ndarray) -> np.ndarray:
    if SKIMAGE_THRESH:
        t = float(threshold_triangle(img_gray))
        return ((img_gray > t).astype(np.uint8) * 255)
    _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    return binary


def mean_binarize(img_gray: np.ndarray) -> np.ndarray:
    t = float(np.mean(img_gray))
    return ((img_gray > t).astype(np.uint8) * 255)


def adaptive_binarize(img_gray: np.ndarray, block_size: int = 15, C: int = 3) -> np.ndarray:
    block_size = ensure_odd(block_size, 3)
    return cv2.adaptiveThreshold(
        img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, int(C)
    )


def adaptive_mean_binarize(img_gray: np.ndarray, block_size: int = 15, C: int = 3) -> np.ndarray:
    block_size = ensure_odd(block_size, 3)
    return cv2.adaptiveThreshold(
        img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, int(C)
    )


def adaptive_gaussian_binarize(img_gray: np.ndarray, block_size: int = 15, C: int = 3) -> np.ndarray:
    return adaptive_binarize(img_gray, block_size=block_size, C=C)


def adaptive_gaussian_binarize_blurred(
    img_gray: np.ndarray,
    blur_ksize: int = 3,
    block_size: int = 21,
    C: int = 2,
) -> np.ndarray:
    """
    Detector-style adaptive Gaussian thresholding:
        median blur -> adaptive Gaussian threshold
    """
    blur_ksize = ensure_odd(blur_ksize, 1)
    block_size = ensure_odd(block_size, 3)
    blurred = median_blur(img_gray, ksize=blur_ksize)
    return cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        int(C),
    )


def adaptive_gaussian_binarize_detector(
    img_gray: np.ndarray,
    blur_ksize: int = 3,
    block_size: int = 21,
    C: int = 2,
    invert: bool = True,
) -> np.ndarray:
    """
    Detector-style binarization:
        median blur -> adaptive Gaussian -> optional invert
    """
    binary = adaptive_gaussian_binarize_blurred(
        img_gray,
        blur_ksize=blur_ksize,
        block_size=block_size,
        C=C,
    )

    if invert:
        binary = cv2.bitwise_not(binary)

    return binary


def sauvola_binarize(img_gray: np.ndarray, window_size: int = 15, k: float = 0.2) -> np.ndarray:
    if SKIMAGE_THRESH:
        t = threshold_sauvola(img_gray, window_size=ensure_odd(window_size, 3), k=float(k))
        return ((img_gray > t).astype(np.uint8) * 255)
    return adaptive_binarize(img_gray, block_size=window_size, C=int(round(k * 10)))


def niblack_binarize(img_gray: np.ndarray, window_size: int = 15, k: float = -0.2) -> np.ndarray:
    if SKIMAGE_THRESH:
        t = threshold_niblack(img_gray, window_size=ensure_odd(window_size, 3), k=float(k))
        return ((img_gray > t).astype(np.uint8) * 255)
    return adaptive_mean_binarize(img_gray, block_size=window_size, C=int(round(-k * 10)))


def invert_binary_if_needed(
    binary: np.ndarray,
    want_ridges_white: bool = True,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    if mask is not None:
        vals = binary[mask > 0]
    else:
        vals = binary.reshape(-1)

    if vals.size == 0:
        return binary

    white_ratio = float(np.mean(vals > 0))

    if want_ridges_white:
        return binary if white_ratio < 0.5 else (255 - binary)
    else:
        return binary if white_ratio > 0.5 else (255 - binary)


def apply_mask_to_binary(binary: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    if mask is None:
        return binary.copy()
    out = binary.copy()
    out[mask == 0] = 0
    return out


def binarize_image(img_gray: np.ndarray, method: str = "adaptive", **kwargs) -> np.ndarray:
    method = method.lower()

    if method == "adaptive":
        return adaptive_binarize(img_gray, kwargs.get("block_size", kwargs.get("block", 15)), kwargs.get("C", 3))
    if method == "adaptive_mean":
        return adaptive_mean_binarize(img_gray, kwargs.get("block_size", kwargs.get("block", 15)), kwargs.get("C", 3))
    if method == "adaptive_gaussian":
        return adaptive_gaussian_binarize(img_gray, kwargs.get("block_size", kwargs.get("block", 15)), kwargs.get("C", 3))
    if method == "fixed":
        return fixed_binarize(img_gray, kwargs.get("thresh", 127))
    if method == "otsu":
        return otsu_binarize(img_gray)
    if method == "li":
        return li_binarize(img_gray)
    if method == "triangle":
        return triangle_binarize(img_gray)
    if method == "mean":
        return mean_binarize(img_gray)
    if method == "sauvola":
        return sauvola_binarize(img_gray, kwargs.get("window_size", 15), kwargs.get("k", 0.2))
    if method == "niblack":
        return niblack_binarize(img_gray, kwargs.get("window_size", 15), kwargs.get("k", -0.2))
    if method == "adaptive_gaussian_blurred":
        return adaptive_gaussian_binarize_blurred(
            img_gray,
            kwargs.get("blur_ksize", 3),
            kwargs.get("block_size", kwargs.get("block", 21)),
            kwargs.get("C", 2),
        )
    if method == "adaptive_gaussian_detector":
        return adaptive_gaussian_binarize_detector(
            img_gray,
            blur_ksize=kwargs.get("blur_ksize", 3),
            block_size=kwargs.get("block_size", kwargs.get("block", 21)),
            C=kwargs.get("C", 2),
            invert=kwargs.get("invert", True),
        )

    raise ValueError(f"Unsupported binarization method: {method}")