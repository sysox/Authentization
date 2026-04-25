from __future__ import annotations

import numpy as np



def rgb_to_gray_luma(img_rgb: np.ndarray) -> np.ndarray:
    r = img_rgb[:, :, 0].astype(np.float32)
    g = img_rgb[:, :, 1].astype(np.float32)
    b = img_rgb[:, :, 2].astype(np.float32)
    return np.clip(0.299 * r + 0.587 * g + 0.114 * b, 0, 255).astype(np.uint8)


def rgb_to_gray_avg(img_rgb: np.ndarray) -> np.ndarray:
    return np.mean(img_rgb.astype(np.float32), axis=2).clip(0, 255).astype(np.uint8)


def rgb_to_gray_max(img_rgb: np.ndarray) -> np.ndarray:
    return np.max(img_rgb, axis=2).astype(np.uint8)


def rgb_to_gray_min(img_rgb: np.ndarray) -> np.ndarray:
    return np.min(img_rgb, axis=2).astype(np.uint8)


def rgb_to_gray_green(img_rgb: np.ndarray) -> np.ndarray:
    return img_rgb[:, :, 1].copy()


def rgb_to_gray_red(img_rgb: np.ndarray) -> np.ndarray:
    return img_rgb[:, :, 0].copy()


def rgb_to_gray_blue(img_rgb: np.ndarray) -> np.ndarray:
    return img_rgb[:, :, 2].copy()


def rgb_to_gray_y(img_rgb: np.ndarray) -> np.ndarray:
    return rgb_to_gray_luma(img_rgb)


def compute_gray_variants(img_rgb: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "gray_default": rgb_to_gray_luma(img_rgb),
        "gray_luma": rgb_to_gray_luma(img_rgb),
        "gray_avg": rgb_to_gray_avg(img_rgb),
        "gray_green": rgb_to_gray_green(img_rgb),
        "gray_red": rgb_to_gray_red(img_rgb),
        "gray_blue": rgb_to_gray_blue(img_rgb),
        "gray_max": rgb_to_gray_max(img_rgb),
        "gray_min": rgb_to_gray_min(img_rgb),
        "gray_y": rgb_to_gray_y(img_rgb),
    }
