from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

try:
    from skimage.morphology import skeletonize
    SKIMAGE_SKELETON = True
except Exception:
    SKIMAGE_SKELETON = False

from .morphology import remove_small_components
from .utils import as_uint8_binary


def get_skeleton_cv(binary_img: np.ndarray, max_iter: int = 100) -> np.ndarray:
    skel = np.zeros(binary_img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    temp_img = binary_img.copy()
    for _ in range(max_iter):
        eroded = cv2.erode(temp_img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(temp_img, temp)
        skel = cv2.bitwise_or(skel, temp)
        temp_img = eroded.copy()
        if cv2.countNonZero(temp_img) == 0:
            break
    return skel


def get_skeleton(
    binary_img: np.ndarray,
    use_skimage: bool = True,
    max_iter: int = 100,
    min_object_size: int = 0,
) -> np.ndarray:
    """
    Compute skeleton from binary ridge image.

    Parameters
    ----------
    binary_img : np.ndarray
        Binary image (ridges=255)
    use_skimage : bool
        Use skimage.skeletonize if available
    max_iter : int
        Iterations for OpenCV fallback
    min_object_size : int
        Remove connected ridge components smaller than this size
        BEFORE skeletonization (helps remove noise fragments)

    Returns
    -------
    skeleton : np.ndarray
        Skeletonized image (uint8, ridges=255)
    """

    bin01 = (binary_img > 0).astype(np.uint8)

    # optional cleanup before skeletonization
    if min_object_size > 0:
        bin01 = remove_small_components(bin01 * 255, min_area=min_object_size)
        bin01 = (bin01 > 0).astype(np.uint8)

    if use_skimage and SKIMAGE_SKELETON:
        skel = skeletonize(bin01 > 0)
        return (skel.astype(np.uint8) * 255)

    return get_skeleton_cv(bin01 * 255, max_iter=max_iter)


def skel_to_bool(skel: np.ndarray) -> np.ndarray:
    return (skel > 0).astype(np.uint8)


def bool_to_skel(img: np.ndarray) -> np.ndarray:
    return as_uint8_binary(img)


def get_8_neighbors(y: int, x: int, h: int, w: int) -> List[Tuple[int, int]]:
    out = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                out.append((ny, nx))
    return out


def count_neighbors(skel01: np.ndarray, y: int, x: int) -> int:
    h, w = skel01.shape
    return sum(int(skel01[ny, nx] > 0) for ny, nx in get_8_neighbors(y, x, h, w))


def classify_skeleton_pixels(skel01: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w = skel01.shape
    endpoints = np.zeros_like(skel01, dtype=np.uint8)
    junctions = np.zeros_like(skel01, dtype=np.uint8)
    ys, xs = np.where(skel01 > 0)
    for y, x in zip(ys, xs):
        n = count_neighbors(skel01, y, x)
        if n == 1:
            endpoints[y, x] = 255
        elif n >= 3:
            junctions[y, x] = 255
    return endpoints, junctions


def trace_branch_from_endpoint(skel01: np.ndarray, start_y: int, start_x: int, max_len: int = 20):
    h, w = skel01.shape
    path = [(start_y, start_x)]
    prev = None
    cur = (start_y, start_x)
    for _ in range(max_len):
        nbrs = [(ny, nx) for ny, nx in get_8_neighbors(cur[0], cur[1], h, w) if skel01[ny, nx] > 0 and (ny, nx) != prev]
        if len(nbrs) == 0:
            break
        if len(nbrs) > 1:
            break
        nxt = nbrs[0]
        path.append(nxt)
        prev, cur = cur, nxt
        if count_neighbors(skel01, cur[0], cur[1]) != 2:
            break
    return path


def remove_short_spurs(skel: np.ndarray, spur_length: int = 8) -> np.ndarray:
    sk = skel_to_bool(skel)
    endpoints, _ = classify_skeleton_pixels(sk)
    ys, xs = np.where(endpoints > 0)
    
    pixels_to_remove = set()
    
    for y, x in zip(ys, xs):
        # We trace on the original 'sk' to avoid topology changes during the pass
        path = trace_branch_from_endpoint(sk, y, x, max_len=spur_length + 2)
        if len(path) <= spur_length:
            end = path[-1]
            n_end = count_neighbors(sk, end[0], end[1])
            if n_end >= 3:
                # Connected to junction. Remove spur, keep junction.
                for py, px in path[:-1]:
                    pixels_to_remove.add((py, px))
            else:
                # Isolated or endpoint-endpoint short ridge
                for py, px in path:
                    pixels_to_remove.add((py, px))

    out = sk.copy()
    for py, px in pixels_to_remove:
        out[py, px] = 0
                    
    return bool_to_skel(out)


def remove_small_skeleton_components(skeleton: np.ndarray, min_size: int = 12) -> np.ndarray:
    return remove_small_components(skeleton, min_area=min_size)


def prune_skeleton_topology(skeleton: np.ndarray, spur_length: int = 8, min_component_size: int = 12,
                            n_passes: int = 3) -> np.ndarray:
    out = skeleton.copy()
    out = remove_small_skeleton_components(out, min_size=min_component_size)
    for _ in range(max(0, n_passes)):
        out = remove_short_spurs(out, spur_length=spur_length)
        out = remove_small_skeleton_components(out, min_size=min_component_size)
    return out