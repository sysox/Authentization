from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np


def _get_8_neighbors_orientation(orient_map: np.ndarray, y: int, x: int) -> List[float]:
    h, w = orient_map.shape
    neighbors = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                neighbors.append(float(orient_map[ny, nx]))
    return neighbors


def compute_poincare_index(orient_map: np.ndarray, y: int, x: int, block_size: int = 3) -> float:
    """
    Computes the Poincare Index at a given point (y, x) in the orientation map.
    The Poincare Index is calculated by summing the changes in orientation
    around a closed path (a square block) centered at (y, x).

    A Poincare Index of:
    - 0.5 indicates a Core point
    - -0.5 indicates a Delta point
    - 0 indicates no singularity
    """
    h, w = orient_map.shape
    half_block = block_size // 2

    # Define the path around the central pixel
    path_coords = []
    # Top row (left to right)
    for i in range(x - half_block, x + half_block + 1):
        path_coords.append((y - half_block, i))
    # Right column (top to bottom)
    for j in range(y - half_block + 1, y + half_block + 1):
        path_coords.append((j, x + half_block))
    # Bottom row (right to left)
    for i in range(x + half_block - 1, x - half_block - 1, -1):
        path_coords.append((y + half_block, i))
    # Left column (bottom to top)
    for j in range(y + half_block - 1, y - half_block, -1):
        path_coords.append((j, x - half_block))

    index = 0.0
    for i in range(len(path_coords)):
        y1, x1 = path_coords[i]
        y2, x2 = path_coords[(i + 1) % len(path_coords)]

        if not (0 <= y1 < h and 0 <= x1 < w and 0 <= y2 < h and 0 <= x2 < w):
            return 0.0  # Path goes out of bounds, cannot compute reliably

        theta1 = orient_map[y1, x1]
        theta2 = orient_map[y2, x2]

        diff = theta2 - theta1
        # Normalize angle difference to be within (-pi/2, pi/2]
        if diff > math.pi / 2:
            diff -= math.pi
        elif diff <= -math.pi / 2:
            diff += math.pi
        index += diff

    # Normalize by pi and round to nearest 0.5
    return round(index / math.pi * 2) / 2.0


def extract_singularities(orient_map: np.ndarray, mask: np.ndarray | None = None, block_size: int = 3) -> List[dict]:
    """
    Extracts singular points (Core and Delta) from an orientation map.
    """
    h, w = orient_map.shape
    singularities = []
    half_block = block_size // 2

    for y in range(half_block, h - half_block):
        for x in range(half_block, w - half_block):
            if mask is not None and mask[y, x] == 0:
                continue

            pi = compute_poincare_index(orient_map, y, x, block_size)

            if abs(pi - 0.5) < 0.1:  # Core point
                singularities.append({"x": x, "y": y, "type": "core", "poincare_index": pi})
            elif abs(pi - (-0.5)) < 0.1:  # Delta point
                singularities.append({"x": x, "y": y, "type": "delta", "poincare_index": pi})

    # Simple clustering to merge nearby singularities
    # This is a basic approach, more advanced clustering might be needed for robustness
    if not singularities:
        return []

    clustered_singularities = []
    for s in singularities:
        found_cluster = False
        for cluster in clustered_singularities:
            # Check if current singularity is close to any existing cluster centroid
            cx = np.mean([cs["x"] for cs in cluster])
            cy = np.mean([cs["y"] for cs in cluster])
            if math.hypot(s["x"] - cx, s["y"] - cy) < block_size and s["type"] == cluster[0]["type"]:
                cluster.append(s)
                found_cluster = True
                break
        if not found_cluster:
            clustered_singularities.append([s])

    final_singularities = []
    for cluster in clustered_singularities:
        avg_x = int(round(np.mean([s["x"] for s in cluster])))
        avg_y = int(round(np.mean([s["y"] for s in cluster])))
        avg_pi = np.mean([s["poincare_index"] for s in cluster])
        final_singularities.append({"x": avg_x, "y": avg_y, "type": cluster[0]["type"], "poincare_index": avg_pi})

    return final_singularities
