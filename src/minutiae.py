from __future__ import annotations

import math
from collections import deque
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .skeleton import get_8_neighbors, skel_to_bool


def crossing_number(skel01: np.ndarray, y: int, x: int) -> int:
    p = [
        skel01[y, x + 1], skel01[y - 1, x + 1], skel01[y - 1, x], skel01[y - 1, x - 1],
        skel01[y, x - 1], skel01[y + 1, x - 1], skel01[y + 1, x], skel01[y + 1, x + 1],
    ]
    s = sum(abs(int(p[i]) - int(p[(i + 1) % 8])) for i in range(8))
    return s // 2


def cluster_points(points: List[dict], dist_thresh: int = 6) -> List[dict]:
    clusters: List[List[dict]] = []
    for p in points:
        placed = False
        for c in clusters:
            cx = np.mean([q["x"] for q in c])
            cy = np.mean([q["y"] for q in c])
            if (p["x"] - cx) ** 2 + (p["y"] - cy) ** 2 <= dist_thresh ** 2 and p["type"] == c[0]["type"]:
                c.append(p)
                placed = True
                break
        if not placed:
            clusters.append([p])

    out = []
    for c in clusters:
        out.append({
            "x": int(round(np.mean([q["x"] for q in c]))),
            "y": int(round(np.mean([q["y"] for q in c]))),
            "type": c[0]["type"],
            "score": len(c),
        })
    return out


def verify_skeleton_polarity(skeleton: np.ndarray, max_white_ratio: float = 0.40) -> bool:
    white_ratio = float(np.count_nonzero(skeleton) / skeleton.size)
    return white_ratio < max_white_ratio


def compute_minutia_orientation(skeleton: np.ndarray, y: int, x: int, radius: int = 10) -> Optional[float]:
    sk = skel_to_bool(skeleton)
    h, w = sk.shape
    pts = []
    for ny in range(max(0, y - radius), min(h, y + radius + 1)):
        for nx in range(max(0, x - radius), min(w, x + radius + 1)):
            if sk[ny, nx] and not (ny == y and nx == x):
                d2 = (nx - x) ** 2 + (ny - y) ** 2
                if d2 <= radius ** 2:
                    pts.append((nx - x, ny - y))
    if len(pts) < 2:
        return None
    arr = np.asarray(pts, dtype=np.float32)
    cov = np.cov(arr.T)
    vals, vecs = np.linalg.eigh(cov)
    v = vecs[:, np.argmax(vals)]
    return float(np.arctan2(v[1], v[0]))


def _trace_branch(skel01: np.ndarray, y: int, x: int, start_y: int, start_x: int, trace_len: int = 10) -> tuple[int, int]:
    h, w = skel01.shape
    prev_y, prev_x = y, x
    cur_y, cur_x = start_y, start_x

    for _ in range(max(0, trace_len - 1)):
        nxt = []
        for ny, nx in get_8_neighbors(cur_y, cur_x, h, w):
            if skel01[ny, nx] > 0 and (ny, nx) != (prev_y, prev_x):
                nxt.append((ny, nx))
        if not nxt:
            break
        prev_y, prev_x = cur_y, cur_x
        cur_y, cur_x = nxt[0]
    return cur_y, cur_x


def compute_minutia_orientation_traced(
    skeleton: np.ndarray,
    y: int,
    x: int,
    minutia_type: str,
    trace_len: int = 10,
) -> Optional[float]:
    sk = skel_to_bool(skeleton)
    h, w = sk.shape

    neighbors = []
    for ny, nx in get_8_neighbors(y, x, h, w):
        if sk[ny, nx] > 0:
            neighbors.append((ny, nx))

    if not neighbors:
        return None

    angles = []
    for ny, nx in neighbors:
        ey, ex = _trace_branch(sk, y, x, ny, nx, trace_len=trace_len)
        angles.append(math.atan2(ey - y, ex - x))

    if minutia_type == "ending":
        return float(angles[0])

    if minutia_type == "bifurcation" and len(angles) >= 3:
        v = [(math.cos(a), math.sin(a)) for a in angles]
        d01 = v[0][0] * v[1][0] + v[0][1] * v[1][1]
        d02 = v[0][0] * v[2][0] + v[0][1] * v[2][1]
        d12 = v[1][0] * v[2][0] + v[1][1] * v[2][1]

        if d01 >= d02 and d01 >= d12:
            stem_idx = 2
        elif d02 >= d01 and d02 >= d12:
            stem_idx = 1
        else:
            stem_idx = 0

        return float((angles[stem_idx] + math.pi) % (2 * math.pi))

    return float(angles[0])


def _bfs_path_length(skel01: np.ndarray, y0: int, x0: int, y1: int, x1: int, max_steps: int = 120) -> int:
    h, w = skel01.shape
    q = deque([(y0, x0, 0)])
    seen = {(y0, x0)}

    while q:
        y, x, d = q.popleft()
        if (y, x) == (y1, x1):
            return d
        if d >= max_steps:
            continue
        for ny, nx in get_8_neighbors(y, x, h, w):
            if (ny, nx) not in seen and skel01[ny, nx] > 0:
                seen.add((ny, nx))
                q.append((ny, nx, d + 1))
    return -1


def _clean_minutiae_spurs_and_short_ridges(
    skeleton: np.ndarray,
    raw: List[dict],
    max_spur_len: int = 15,
) -> List[dict]:
    sk = skel_to_bool(skeleton)

    endings = [m for m in raw if m["type"] == "ending"]
    bifurs = [m for m in raw if m["type"] == "bifurcation"]

    e_remove = [False] * len(endings)
    b_remove = [False] * len(bifurs)

    # ending <-> bifurcation spur
    for i, e in enumerate(endings):
        if e_remove[i]:
            continue
        for j, b in enumerate(bifurs):
            if b_remove[j]:
                continue
            dist = math.hypot(e["x"] - b["x"], e["y"] - b["y"])
            if dist < max_spur_len:
                p_len = _bfs_path_length(sk, e["y"], e["x"], b["y"], b["x"], max_steps=max_spur_len)
                if 0 < p_len <= max_spur_len:
                    e_remove[i] = True
                    b_remove[j] = True
                    break

    # ending <-> ending short ridge
    max_ridge_len = max_spur_len + 5
    for i in range(len(endings)):
        if e_remove[i]:
            continue
        e1 = endings[i]
        for j in range(i + 1, len(endings)):
            if e_remove[j]:
                continue
            e2 = endings[j]
            dist = math.hypot(e1["x"] - e2["x"], e1["y"] - e2["y"])
            if dist < max_ridge_len:
                p_len = _bfs_path_length(sk, e1["y"], e1["x"], e2["y"], e2["x"], max_steps=max_ridge_len)
                if 0 < p_len <= max_ridge_len:
                    e_remove[i] = True
                    e_remove[j] = True
                    break

    out = []
    for i, e in enumerate(endings):
        if not e_remove[i]:
            out.append(e)
    for j, b in enumerate(bifurs):
        if not b_remove[j]:
            out.append(b)
    return out


def suppress_close_minutiae(points: List[dict], min_dist: int = 12) -> List[dict]:
    if not points:
        return []

    kept = []
    for p in sorted(points, key=lambda q: -float(q.get("confidence", q.get("score", 1.0)))):
        too_close = False
        for k in kept:
            if p["type"] == k["type"]:
                if math.hypot(p["x"] - k["x"], p["y"] - k["y"]) < min_dist:
                    too_close = True
                    break
        if not too_close:
            kept.append(p)
    return kept


def extract_minutiae_crossing_number(
    skeleton: np.ndarray,
    mask: np.ndarray | None = None,
    border_margin: int = 10,
    cluster_dist: int = 6,
    swap_on_inverted_polarity: bool = True,
    use_traced_orientation: bool = True,
) -> List[dict]:
    sk = skel_to_bool(skeleton)
    h, w = sk.shape
    pts = []

    polarity_ok = verify_skeleton_polarity(skeleton)

    for y in range(1 + border_margin, h - 1 - border_margin):
        for x in range(1 + border_margin, w - 1 - border_margin):
            if not sk[y, x]:
                continue
            if mask is not None and mask[y, x] == 0:
                continue

            cn = crossing_number(sk, y, x)

            if cn == 1:
                mtype = "ending"
                if swap_on_inverted_polarity and not polarity_ok:
                    mtype = "bifurcation"
                pts.append({"x": x, "y": y, "type": mtype, "confidence": 0.85})

            elif cn == 3:
                mtype = "bifurcation"
                if swap_on_inverted_polarity and not polarity_ok:
                    mtype = "ending"
                pts.append({"x": x, "y": y, "type": mtype, "confidence": 0.88})

    merged = cluster_points(pts, dist_thresh=cluster_dist)

    for m in merged:
        if use_traced_orientation:
            ang = compute_minutia_orientation_traced(skeleton, m["y"], m["x"], m["type"])
        else:
            ang = compute_minutia_orientation(skeleton, m["y"], m["x"])

        m["angle"] = ang
        m["source"] = "auto"

    return merged


def finalize_minutiae(
    raw: List[dict],
    skeleton: np.ndarray,
    max_spur_len: int = 15,
    suppress_dist: int = 12,
) -> List[dict]:
    cleaned = _clean_minutiae_spurs_and_short_ridges(
        skeleton,
        raw,
        max_spur_len=max_spur_len,
    )
    cleaned = suppress_close_minutiae(cleaned, min_dist=suppress_dist)

    for i, m in enumerate(cleaned, start=1):
        m["id"] = i
    return cleaned


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

# BGR colours — high saturation, easy to distinguish on any background
_COLOR_ENDING       = (0,   0,   220)  # red (BGR)
_COLOR_BIFURCATION  = (220,  60,   0)  # blue (BGR)
_COLOR_OUTLINE      = (255, 255, 255)  # white outline for contrast on dark bg
_COLOR_OUTLINE_DARK = (30,   30,  30)  # dark outline for contrast on light bg


def _make_canvas(
    background: Optional[np.ndarray],
    canvas_size: Optional[Tuple[int, int]],
) -> np.ndarray:
    """Return a BGR canvas from a background image or a black canvas."""
    if background is not None:
        img = np.asarray(background)
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.ndim == 3 and img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img.copy()

    if canvas_size is None:
        raise ValueError("Provide background or canvas_size=(height, width).")
    h, w = int(canvas_size[0]), int(canvas_size[1])
    return np.zeros((h, w, 3), dtype=np.uint8)


def _draw_marker(
    canvas: np.ndarray,
    x: int, y: int,
    color: tuple,
    radius: int,
    arrow_len: int,
    angle: Optional[float],
    show_arrows: bool = False,
) -> None:
    """Center dot + circle ring around it, same style for all minutiae types."""
    # 1-pixel center dot
    cv2.circle(canvas, (x, y), 1, color, -1, cv2.LINE_AA)
    # circle ring
    cv2.circle(canvas, (x, y), radius, color, 1, cv2.LINE_AA)

    if show_arrows and angle is not None:
        ex = int(x + arrow_len * math.cos(angle))
        ey = int(y + arrow_len * math.sin(angle))
        cv2.arrowedLine(canvas, (x, y), (ex, ey), color, 1, cv2.LINE_AA, tipLength=0.30)


def _draw_marker_ending(canvas, x, y, radius, arrow_len, angle, show_arrows=False):
    _draw_marker(canvas, x, y, _COLOR_ENDING, radius, arrow_len, angle, show_arrows)


def _draw_marker_bifurcation(canvas, x, y, radius, arrow_len, angle, show_arrows=False):
    _draw_marker(canvas, x, y, _COLOR_BIFURCATION, radius, arrow_len, angle, show_arrows)


def _is_dark_region(canvas: np.ndarray, x: int, y: int, patch: int = 12) -> bool:
    """True if the neighbourhood around (x, y) is mostly dark."""
    h, w = canvas.shape[:2]
    y0, y1 = max(0, y - patch), min(h, y + patch)
    x0, x1 = max(0, x - patch), min(w, x + patch)
    region = canvas[y0:y1, x0:x1]
    return float(np.mean(region)) < 128


def _draw_legend(
    canvas: np.ndarray,
    n_endings: int,
    n_bifurcations: int,
    radius: int,
) -> None:
    h, w = canvas.shape[:2]
    pad = 10
    box_w, box_h = 210, 70
    x0, y0 = pad, h - box_h - pad

    # Semi-transparent dark panel
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, canvas, 0.35, 0, canvas)

    r = radius

    # Ending row
    ex, ey = x0 + r + 6, y0 + 18
    cv2.circle(canvas, (ex, ey), 1,  _COLOR_ENDING, -1, cv2.LINE_AA)
    cv2.circle(canvas, (ex, ey), r,  _COLOR_ENDING,  1, cv2.LINE_AA)
    cv2.putText(canvas, f"Ending  ({n_endings})",
                (ex + r + 8, ey + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (230, 230, 230), 1, cv2.LINE_AA)

    # Bifurcation row
    bx, by = x0 + r + 6, y0 + 50
    cv2.circle(canvas, (bx, by), 1,  _COLOR_BIFURCATION, -1, cv2.LINE_AA)
    cv2.circle(canvas, (bx, by), r,  _COLOR_BIFURCATION,  1, cv2.LINE_AA)
    cv2.putText(canvas, f"Bifurcation  ({n_bifurcations})",
                (bx + r + 8, by + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (230, 230, 230), 1, cv2.LINE_AA)


def draw_minutiae(
    minutiae: List[dict],
    background: Optional[np.ndarray] = None,
    *,
    canvas_size: Optional[Tuple[int, int]] = None,
    show_type: str = "both",
    show_arrows: bool = False,
    show_legend: bool = True,
) -> np.ndarray:
    """
    Render minutiae with large, human-readable markers on a chosen background.

    Parameters
    ----------
    minutiae : list of minutia dicts
        Output of ``extract_minutiae_crossing_number`` / ``finalize_minutiae``.
    background : np.ndarray or None
        What to draw on top of.  Pass any of:
          - the **original** fingerprint image
          - the **skeleton** image
          - a **binary** image
          - ``None``  →  black canvas (requires *canvas_size*)
    canvas_size : (height, width)
        Size of the black canvas when *background* is ``None``.
        Ignored when a background image is given.
    radius : int
        Marker half-size in pixels (default 10).
    arrow_len : int
        Length of the orientation arrow in pixels (default 24).
    show_legend : bool
        Draw a small legend with counts in the bottom-left corner.

    Returns
    -------
    np.ndarray
        BGR image ready for display or saving.

    Shape legend
    ------------
    Filled **circle**   (red-orange) = ridge ending
    Filled **diamond**  (cyan-blue)  = bifurcation
    White arrow from centre          = minutia orientation

    Examples
    --------
    >>> vis = draw_minutiae(minutiae, background=original_img)
    >>> vis = draw_minutiae(minutiae, background=skeleton)
    >>> vis = draw_minutiae(minutiae, canvas_size=skeleton.shape[:2])  # black bg

    Display in a notebook::

        import matplotlib.pyplot as plt, cv2
        plt.figure(figsize=(8, 8))
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
    """
    _RADIUS    = 5   # fixed marker size — not user-adjustable
    _ARROW_LEN = 18  # fixed arrow length

    canvas = _make_canvas(background, canvas_size)
    n_endings = n_bifurcations = 0
    show_endings      = show_type in ("both", "endings")
    show_bifurcations = show_type in ("both", "bifurcations")

    for m in minutiae:
        x = int(round(m["x"]))
        y = int(round(m["y"]))
        h, w = canvas.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            continue

        angle = m.get("angle")
        mtype = m.get("type", "ending")

        if mtype == "ending" and show_endings:
            _draw_marker_ending(canvas, x, y, _RADIUS, _ARROW_LEN, angle, show_arrows)
            n_endings += 1
        elif mtype != "ending" and show_bifurcations:
            _draw_marker_bifurcation(canvas, x, y, _RADIUS, _ARROW_LEN, angle, show_arrows)
            n_bifurcations += 1

    if show_legend:
        _draw_legend(canvas, n_endings, n_bifurcations, _RADIUS)

    return canvas