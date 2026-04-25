"""
Minutiae-based fingerprint matching.

Algorithm
---------
1. **Alignment search** – for every same-type seed pair (a_i, b_j) compute the
   rigid transform (rotation + translation) that maps b_j onto a_i. Apply that
   transform to the whole set B, then greedily count how many minutiae fall
   within (dist_tol, angle_tol). Keep the transform with the most matches.

2. **Match extraction** – return the 1-to-1 matched pairs found under the best
   transform, together with a similarity score in [0, 1].

Public API
----------
align_minutiae(set_a, set_b, ...)      → (matched_pairs, transform)
match_score(set_a, set_b, pairs)       → float
visualize_minutiae_match(...)          → np.ndarray   (side-by-side image)
compare_fingerprints(img_a, min_a, img_b, min_b, ...)
                                       → (result_image, matched_pairs, score)
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import cv2
import numpy as np

# A minutia is a dict with keys: x, y, type, angle (may be None), id, ...
Minutia = dict
MatchedPairs = List[Tuple[int, int]]
Transform = Tuple[float, float, float]  # (dx, dy, dtheta)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _angle_diff(a1: float, a2: float) -> float:
    """Smallest absolute difference between two angles, result in [0, π]."""
    d = abs(a1 - a2) % (2 * math.pi)
    return d if d <= math.pi else 2 * math.pi - d


def _apply_rigid(
    points: List[Minutia],
    dx: float,
    dy: float,
    dtheta: float,
) -> List[dict]:
    """
    Return a new list where each point has been rotated by *dtheta* (around the
    origin) and then translated by *(dx, dy)*.  Original dicts are not mutated.
    """
    cos_t = math.cos(dtheta)
    sin_t = math.sin(dtheta)
    out = []
    for m in points:
        rx = m["x"] * cos_t - m["y"] * sin_t + dx
        ry = m["x"] * sin_t + m["y"] * cos_t + dy
        ang = m.get("angle")
        ra = None if ang is None else (ang + dtheta) % (2 * math.pi)
        out.append({**m, "x": rx, "y": ry, "angle": ra})
    return out


# ---------------------------------------------------------------------------
# Core matching logic
# ---------------------------------------------------------------------------

def _greedy_match(
    set_a: List[Minutia],
    set_b_transformed: List[Minutia],
    dist_tol: float,
    angle_tol: float,
) -> MatchedPairs:
    """
    1-to-1 greedy matching between set_a and an already-transformed set_b.
    Candidates are sorted by distance so closer pairs are preferred.
    """
    candidates: List[Tuple[float, int, int]] = []

    for i, a in enumerate(set_a):
        for j, bt in enumerate(set_b_transformed):
            if a["type"] != bt["type"]:
                continue
            d = math.hypot(a["x"] - bt["x"], a["y"] - bt["y"])
            if d > dist_tol:
                continue
            ang_a = a.get("angle")
            ang_b = bt.get("angle")
            if ang_a is not None and ang_b is not None:
                if _angle_diff(ang_a, ang_b) > angle_tol:
                    continue
            candidates.append((d, i, j))

    candidates.sort()
    used_a: set[int] = set()
    used_b: set[int] = set()
    pairs: MatchedPairs = []
    for _, i, j in candidates:
        if i not in used_a and j not in used_b:
            pairs.append((i, j))
            used_a.add(i)
            used_b.add(j)
    return pairs


def align_minutiae(
    set_a: List[Minutia],
    set_b: List[Minutia],
    dist_tol: float = 20.0,
    angle_tol: float = 0.35,
    max_seeds: int = 0,
) -> Tuple[MatchedPairs, Optional[Transform]]:
    """
    Find the rigid transform (rotation + translation) that best aligns *set_b*
    to *set_a*, then return the matched minutiae pairs.

    Parameters
    ----------
    set_a, set_b : list of minutia dicts
        Output of ``extract_minutiae_crossing_number`` / ``finalize_minutiae``.
    dist_tol : float
        Maximum spatial distance (pixels) for two minutiae to be considered a
        match after alignment.
    angle_tol : float
        Maximum angle difference (radians) for a match.
    max_seeds : int
        Limit the number of seed pairs tried (0 = try all). Reduces runtime on
        large sets at the cost of possibly missing the optimal alignment.

    Returns
    -------
    matched_pairs : list of (idx_a, idx_b)
        Positional indices into *set_a* and *set_b*.
    transform : (dx, dy, dtheta) or None
        Best rigid transform: rotate set_b by *dtheta*, then translate by
        *(dx, dy)*.  None if no alignment could be found.
    """
    best_pairs: MatchedPairs = []
    best_transform: Optional[Transform] = None

    seeds_tried = 0
    for i, a in enumerate(set_a):
        a_ang = a.get("angle")
        if a_ang is None:
            continue
        for j, b in enumerate(set_b):
            if a["type"] != b["type"]:
                continue
            b_ang = b.get("angle")
            if b_ang is None:
                continue

            dtheta = a_ang - b_ang
            cos_t = math.cos(dtheta)
            sin_t = math.sin(dtheta)
            # rotated position of b
            rx = b["x"] * cos_t - b["y"] * sin_t
            ry = b["x"] * sin_t + b["y"] * cos_t
            dx = a["x"] - rx
            dy = a["y"] - ry

            b_transformed = _apply_rigid(set_b, dx, dy, dtheta)
            pairs = _greedy_match(set_a, b_transformed, dist_tol, angle_tol)

            if len(pairs) > len(best_pairs):
                best_pairs = pairs
                best_transform = (dx, dy, dtheta)

            seeds_tried += 1
            if max_seeds > 0 and seeds_tried >= max_seeds:
                return best_pairs, best_transform

    # Fallback: try identity (images already roughly aligned)
    if not best_pairs:
        pairs = _greedy_match(set_a, set_b, dist_tol, angle_tol)
        if pairs:
            best_pairs = pairs
            best_transform = (0.0, 0.0, 0.0)

    return best_pairs, best_transform


def match_score(
    set_a: List[Minutia],
    set_b: List[Minutia],
    matched_pairs: MatchedPairs,
) -> float:
    """
    Similarity score in [0, 1].

    Defined as ``|matches| / max(|A|, |B|)``.  A score of 1.0 means every
    minutia in the larger set was matched.
    """
    return len(matched_pairs) / max(len(set_a), len(set_b), 1)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

# Colour palette (BGR for OpenCV)
_GREEN  = (50,  210,  50)   # matched minutiae
_RED    = (60,   60, 210)   # unmatched endings
_BLUE   = (210,  60,  60)   # unmatched bifurcations
_YELLOW = (0,   220, 220)   # match lines (yellow in BGR)
_GREY   = (130, 130, 130)   # separator


def _to_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img.copy()


def _draw_minutiae_on_canvas(
    canvas: np.ndarray,
    minutiae: List[Minutia],
    matched_indices: set,
    arrow_len: int = 12,
) -> None:
    """Draw minutiae circles + orientation arrows in-place."""
    for k, m in enumerate(minutiae):
        x, y = int(round(m["x"])), int(round(m["y"]))
        if k in matched_indices:
            color  = _GREEN
            radius = 7
            thick  = 2
        else:
            color  = _RED if m["type"] == "ending" else _BLUE
            radius = 5
            thick  = 1

        cv2.circle(canvas, (x, y), radius, color, thick, cv2.LINE_AA)

        ang = m.get("angle")
        if ang is not None:
            ex = int(x + arrow_len * math.cos(ang))
            ey = int(y + arrow_len * math.sin(ang))
            cv2.arrowedLine(canvas, (x, y), (ex, ey), color, 1,
                            cv2.LINE_AA, tipLength=0.35)


def _add_label(canvas: np.ndarray, text: str, pos: Tuple[int, int]) -> None:
    cv2.putText(
        canvas, text, pos,
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA,
    )
    cv2.putText(
        canvas, text, pos,
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA,
    )


def visualize_minutiae_match(
    img_a: np.ndarray,
    min_a: List[Minutia],
    img_b: np.ndarray,
    min_b: List[Minutia],
    matched_pairs: MatchedPairs,
    label_a: str = "Image A",
    label_b: str = "Image B",
    line_alpha: float = 0.55,
) -> np.ndarray:
    """
    Produce a side-by-side comparison image showing matched minutiae.

    Legend
    ------
    Green  circles  — matched minutiae (both images)
    Red    circles  — unmatched endings
    Blue   circles  — unmatched bifurcations
    Yellow lines    — correspondences drawn horizontally across the two images

    Parameters
    ----------
    img_a, img_b : np.ndarray
        Fingerprint images (grayscale or BGR/RGB).
    min_a, min_b : list of minutia dicts
        Minutiae extracted from each image (positional indices must match
        those in *matched_pairs*).
    matched_pairs : list of (idx_a, idx_b)
        Output of :func:`align_minutiae`.
    label_a, label_b : str
        Caption drawn at the top of each half.
    line_alpha : float
        Opacity of the match lines blended over the image (0–1).

    Returns
    -------
    np.ndarray
        Combined BGR image.
    """
    canvas_a = _to_rgb(img_a)
    canvas_b = _to_rgb(img_b)

    matched_a = {i for i, _ in matched_pairs}
    matched_b = {j for _, j in matched_pairs}

    _draw_minutiae_on_canvas(canvas_a, min_a, matched_a)
    _draw_minutiae_on_canvas(canvas_b, min_b, matched_b)

    # --- pad to equal height ---
    h_a, w_a = canvas_a.shape[:2]
    h_b, w_b = canvas_b.shape[:2]
    h = max(h_a, h_b)

    def _pad(img: np.ndarray, target_h: int) -> np.ndarray:
        dh = target_h - img.shape[0]
        if dh <= 0:
            return img
        pad = np.zeros((dh, img.shape[1], 3), dtype=np.uint8)
        return np.vstack([img, pad])

    canvas_a = _pad(canvas_a, h)
    canvas_b = _pad(canvas_b, h)

    sep_w = 4
    separator = np.full((h, sep_w, 3), 80, dtype=np.uint8)
    combined = np.hstack([canvas_a, separator, canvas_b])

    # --- draw match lines on a separate layer for alpha blending ---
    line_layer = combined.copy()
    offset_x = w_a + sep_w

    for idx_a, idx_b in matched_pairs:
        xa = int(round(min_a[idx_a]["x"]))
        ya = int(round(min_a[idx_a]["y"]))
        xb = int(round(min_b[idx_b]["x"])) + offset_x
        yb = int(round(min_b[idx_b]["y"]))
        cv2.line(line_layer, (xa, ya), (xb, yb), _YELLOW, 2, cv2.LINE_AA)

    cv2.addWeighted(line_layer, line_alpha, combined, 1.0 - line_alpha, 0, combined)

    # --- labels and score ---
    n_match = len(matched_pairs)
    score = match_score(min_a, min_b, matched_pairs)

    _add_label(combined, label_a, (8, 22))
    _add_label(combined, label_b, (offset_x + 8, 22))
    _add_label(
        combined,
        f"matches: {n_match}  |  score: {score:.3f}",
        (offset_x // 2 - 80, h - 10),
    )

    return combined


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------

def compare_fingerprints(
    img_a: np.ndarray,
    min_a: List[Minutia],
    img_b: np.ndarray,
    min_b: List[Minutia],
    dist_tol: float = 20.0,
    angle_tol: float = 0.35,
    max_seeds: int = 0,
    label_a: str = "Probe",
    label_b: str = "Reference",
) -> Tuple[np.ndarray, MatchedPairs, float]:
    """
    Align, match, and visualise two fingerprint minutiae sets in one call.

    Parameters
    ----------
    img_a, img_b : np.ndarray
        Processed fingerprint images (used only for display).
    min_a, min_b : list of minutia dicts
        Minutiae from each image.
    dist_tol : float
        Spatial tolerance in pixels for a match.
    angle_tol : float
        Angular tolerance in radians for a match.
    max_seeds : int
        Limit alignment seeds (0 = exhaustive).
    label_a, label_b : str
        Captions on the result image.

    Returns
    -------
    result_img : np.ndarray
        Side-by-side comparison (BGR).
    matched_pairs : list of (idx_a, idx_b)
    score : float
        Similarity score in [0, 1].

    Example
    -------
    >>> result, pairs, score = compare_fingerprints(img1, min1, img2, min2)
    >>> print(f"Score: {score:.3f}  |  Matched: {len(pairs)}")
    >>> # display in notebook:
    >>> import matplotlib.pyplot as plt
    >>> plt.figure(figsize=(14, 7))
    >>> plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    >>> plt.axis('off'); plt.show()
    """
    pairs, transform = align_minutiae(
        min_a, min_b,
        dist_tol=dist_tol,
        angle_tol=angle_tol,
        max_seeds=max_seeds,
    )
    score = match_score(min_a, min_b, pairs)
    result = visualize_minutiae_match(
        img_a, min_a, img_b, min_b, pairs,
        label_a=label_a, label_b=label_b,
    )
    return result, pairs, score