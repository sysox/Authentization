"""
Scale calibration for fingerprint images.

The user clicks two points on the image and enters the real-world distance
between them. The image is then rescaled so that printing at a given DPI
produces the correct physical dimensions.

Requires an interactive matplotlib backend in Jupyter:
    %matplotlib widget          # preferred (needs ipympl: pip install ipympl)
    %matplotlib notebook        # older alternative
"""
from __future__ import annotations

import math
from typing import Optional

import cv2
import ipywidgets as widgets
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------

def pixel_distance(p1: tuple, p2: tuple) -> float:
    """Euclidean distance between two (x, y) points in pixels."""
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def rescale_to_print_dpi(
    image: np.ndarray,
    px_per_mm: float,
    dpi: int = 300,
) -> np.ndarray:
    """
    Resize *image* so that printing at *dpi* reproduces the correct physical size.

    Parameters
    ----------
    image : np.ndarray
        Source image (uint8 grayscale or RGB).
    px_per_mm : float
        Current resolution of the image in pixels per millimetre,
        measured by the calibration step.
    dpi : int
        Target print resolution (dots per inch). Typical values: 150, 300, 600.

    Returns
    -------
    np.ndarray
        Resized image.
    """
    if px_per_mm <= 0:
        raise ValueError(f"px_per_mm must be positive, got {px_per_mm}")

    target_px_per_mm = dpi / 25.4          # 1 inch = 25.4 mm
    scale = target_px_per_mm / px_per_mm

    h, w = image.shape[:2]
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    # Downscale: LANCZOS4 preserves fine ridge detail better
    # Upscale:   LINEAR avoids ringing artefacts on binary/skeleton images
    interp = cv2.INTER_LANCZOS4 if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(image, (new_w, new_h), interpolation=interp)


def print_size_mm(image: np.ndarray, px_per_mm: float) -> tuple[float, float]:
    """Return (width_mm, height_mm) at the given resolution."""
    h, w = image.shape[:2]
    return w / px_per_mm, h / px_per_mm


# ---------------------------------------------------------------------------
# Interactive calibrator
# ---------------------------------------------------------------------------

class ScaleCalibrator:
    """
    Interactive Jupyter widget for scale calibration.

    Usage
    -----
    >>> # In a notebook cell, first set an interactive backend:
    >>> # %matplotlib widget
    >>> cal = ScaleCalibrator(image)
    >>> cal.display()
    >>> # … click two points, enter distance, press Apply …
    >>> scaled = cal.scaled_image   # rescaled result
    >>> ppm    = cal.px_per_mm      # measured resolution

    Parameters
    ----------
    image : np.ndarray
        Fingerprint image (grayscale or RGB, uint8).
    default_dpi : int
        Default target print DPI shown in the UI (default 300).
    figsize : tuple
        Matplotlib figure size in inches.
    """

    def __init__(
        self,
        image: np.ndarray,
        default_dpi: int = 300,
        figsize: tuple = (8, 8),
    ):
        self.original: np.ndarray = image.copy()
        self.scaled_image: Optional[np.ndarray] = None
        self.px_per_mm: Optional[float] = None

        self._dpi = default_dpi
        self._figsize = figsize

        self._points: list[tuple[float, float]] = []
        self._px_distance: Optional[float] = None

        # matplotlib handles – set in display()
        self._fig = None
        self._ax = None
        self._cid = None

        # widget handles – set in display()
        self._status_lbl: Optional[widgets.Label] = None
        self._dist_input: Optional[widgets.FloatText] = None
        self._dpi_drop: Optional[widgets.Dropdown] = None
        self._apply_btn: Optional[widgets.Button] = None
        self._reset_btn: Optional[widgets.Button] = None
        self._result_html: Optional[widgets.HTML] = None

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def display(self) -> None:
        """Render the calibration UI in the current Jupyter cell."""
        self._build_figure()
        self._build_controls()

        plt.tight_layout()
        plt.show()
        display(self._controls_box)

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------

    def _build_figure(self) -> None:
        self._fig, self._ax = plt.subplots(figsize=self._figsize)
        self._draw_image()
        self._cid = self._fig.canvas.mpl_connect(
            "button_press_event", self._on_click
        )

    def _draw_image(self) -> None:
        self._ax.cla()
        if self.original.ndim == 2:
            self._ax.imshow(self.original, cmap="gray", interpolation="nearest")
        else:
            self._ax.imshow(self.original, interpolation="nearest")
        self._ax.set_title(
            "Click two points whose real-world distance you know",
            fontsize=11,
        )
        self._ax.axis("off")

    def _redraw_annotations(self) -> None:
        """Re-draw point markers and connecting line after a reset."""
        self._draw_image()
        for i, (x, y) in enumerate(self._points):
            self._ax.plot(x, y, "ro", markersize=9, zorder=5)
            self._ax.text(
                x + 6, y - 6, str(i + 1),
                color="red", fontsize=10, fontweight="bold", zorder=6,
            )
        if len(self._points) == 2:
            self._draw_measurement_line()
        self._fig.canvas.draw_idle()

    def _draw_measurement_line(self) -> None:
        x0, y0 = self._points[0]
        x1, y1 = self._points[1]
        self._ax.plot([x0, x1], [y0, y1], "r-", linewidth=2, zorder=4)
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        self._ax.text(
            mx, my,
            f"  {self._px_distance:.1f} px",
            color="red", fontsize=11, fontweight="bold", zorder=6,
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2),
        )

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------

    def _build_controls(self) -> None:
        self._status_lbl = widgets.Label(
            value="Step 1 — click the first point on the image"
        )
        self._dist_input = widgets.FloatText(
            value=10.0,
            min=0.01,
            step=0.5,
            description="Distance (mm):",
            style={"description_width": "130px"},
            layout=widgets.Layout(width="220px"),
        )
        self._dpi_drop = widgets.Dropdown(
            options=[72, 150, 300, 600],
            value=self._dpi,
            description="Print DPI:",
            style={"description_width": "90px"},
            layout=widgets.Layout(width="180px"),
        )
        self._apply_btn = widgets.Button(
            description="Apply Scale",
            button_style="primary",
            disabled=True,
            layout=widgets.Layout(width="120px"),
        )
        self._reset_btn = widgets.Button(
            description="Reset Points",
            button_style="warning",
            layout=widgets.Layout(width="120px"),
        )
        self._result_html = widgets.HTML(value="")

        self._apply_btn.on_click(self._on_apply)
        self._reset_btn.on_click(self._on_reset)
        self._dpi_drop.observe(self._on_dpi_change, names="value")

        self._controls_box = widgets.VBox([
            self._status_lbl,
            widgets.HBox([self._dist_input, self._dpi_drop]),
            widgets.HBox([self._apply_btn, self._reset_btn]),
            self._result_html,
        ])

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_click(self, event) -> None:
        if event.inaxes != self._ax:
            return
        if len(self._points) >= 2:
            return
        if event.xdata is None or event.ydata is None:
            return

        pt = (event.xdata, event.ydata)
        self._points.append(pt)
        idx = len(self._points)

        self._ax.plot(pt[0], pt[1], "ro", markersize=9, zorder=5)
        self._ax.text(
            pt[0] + 6, pt[1] - 6, str(idx),
            color="red", fontsize=10, fontweight="bold", zorder=6,
        )

        if idx == 1:
            self._status_lbl.value = "Step 2 — click the second point"
        else:
            self._px_distance = pixel_distance(self._points[0], self._points[1])
            self._draw_measurement_line()
            self._apply_btn.disabled = False
            self._status_lbl.value = (
                f"Pixel distance: {self._px_distance:.1f} px  —  "
                "enter the real distance and press Apply"
            )

        self._fig.canvas.draw_idle()

    def _on_apply(self, _btn) -> None:
        real_mm = self._dist_input.value
        if real_mm <= 0:
            self._result_html.value = (
                "<span style='color:red'>Distance must be greater than 0.</span>"
            )
            return

        self.px_per_mm = self._px_distance / real_mm
        self._dpi = self._dpi_drop.value
        self.scaled_image = rescale_to_print_dpi(self.original, self.px_per_mm, self._dpi)

        self._update_result_label(real_mm)

    def _on_reset(self, _btn) -> None:
        self._points = []
        self._px_distance = None
        self._apply_btn.disabled = True
        self._status_lbl.value = "Step 1 — click the first point on the image"
        self._result_html.value = ""
        self._redraw_annotations()

    def _on_dpi_change(self, change) -> None:
        # Re-apply with new DPI if calibration already done
        if self.px_per_mm is not None:
            self._dpi = change["new"]
            self.scaled_image = rescale_to_print_dpi(
                self.original, self.px_per_mm, self._dpi
            )
            self._update_result_label(self._dist_input.value)

    # ------------------------------------------------------------------
    # Result display
    # ------------------------------------------------------------------

    def _update_result_label(self, real_mm: float) -> None:
        target_px_per_mm = self._dpi / 25.4
        scale = target_px_per_mm / self.px_per_mm

        orig_h, orig_w = self.original.shape[:2]
        new_h, new_w = self.scaled_image.shape[:2]

        orig_w_mm, orig_h_mm = print_size_mm(self.original, self.px_per_mm)
        new_w_mm = new_w / target_px_per_mm
        new_h_mm = new_h / target_px_per_mm

        self._result_html.value = (
            "<table style='border-collapse:collapse;font-size:13px'>"
            f"<tr><td style='padding:2px 10px 2px 0'><b>Measured scale</b></td>"
            f"<td>{self.px_per_mm:.3f} px/mm</td></tr>"
            f"<tr><td><b>Scale factor</b></td>"
            f"<td>{scale:.4f}×  ({'+' if scale>=1 else ''}{(scale-1)*100:.1f}%)</td></tr>"
            f"<tr><td><b>Original size</b></td>"
            f"<td>{orig_w}×{orig_h} px &nbsp;→&nbsp; "
            f"{orig_w_mm:.1f}×{orig_h_mm:.1f} mm</td></tr>"
            f"<tr><td><b>Rescaled size</b></td>"
            f"<td>{new_w}×{new_h} px</td></tr>"
            f"<tr><td><b>Print size @ {self._dpi} DPI</b></td>"
            f"<td>{new_w_mm:.1f}×{new_h_mm:.1f} mm &nbsp; "
            f"({new_w_mm/25.4:.2f}×{new_h_mm/25.4:.2f} in)</td></tr>"
            "</table>"
        )


# ---------------------------------------------------------------------------
# Widget-based calibrator — no ipyevents, works in PyCharm and all Jupyter
# ---------------------------------------------------------------------------

import io as _io

from PIL import Image as _PILImage, ImageDraw as _ImageDraw

_DISPLAY_MAX = 800          # preview image (longer side)
_GRID_MAX_W  = 200          # result grid image — ~4× smaller than preview
_GRID_MAX_H  = 270          # result grid image — ~3× smaller than preview


def _render_overlay(bg_arr: np.ndarray, pts_img: list, scale: float) -> bytes:
    """Draw calibration points + line on a pre-scaled background array."""
    pil = _PILImage.fromarray(bg_arr)
    draw = _ImageDraw.Draw(pil)
    canvas = [(int(x * scale), int(y * scale)) for x, y in pts_img]
    colors = ["red", "blue"]
    for i, (cx, cy) in enumerate(canvas):
        r = 8
        draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                     fill=colors[i], outline="white")
        draw.text((cx + 10, cy - 13), str(i + 1), fill=colors[i])
    if len(canvas) == 2:
        (dx0, dy0), (dx1, dy1) = canvas
        draw.line([dx0, dy0, dx1, dy1], fill="red", width=2)
        mx, my = (dx0 + dx1) // 2, (dy0 + dy1) // 2
        dist = pixel_distance(pts_img[0], pts_img[1])
        draw.text((mx + 5, my - 14), f"{dist:.1f} px", fill="yellow")
    buf = _io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


class ScaleCalibratorIPY:
    """
    Pure-ipywidgets scale calibrator — no ipyevents, no DOM events.
    Works in PyCharm, JupyterLab, classic Jupyter, VS Code, Colab.

    Workflow
    --------
    1. The image is shown — hover over it to read pixel coordinates from the
       axis labels (matplotlib) or status bar.
    2. Enter the pixel coordinates of two reference points using the sliders.
    3. Set the real-world distance between them (mm) and target print DPI.
    4. Press **Apply Scale**.

    After Apply:
        cal.px_per_mm  – measured resolution
        cal._dpi       – selected target DPI
    """

    def __init__(self, image: np.ndarray, default_dpi: int = 300):
        self.image = image
        self.px_per_mm: Optional[float] = None
        self._dpi = default_dpi

        h, w = image.shape[:2]
        # Scale so the longer side equals _DISPLAY_MAX (upscaling allowed)
        scale = _DISPLAY_MAX / max(w, h)
        cw = max(1, int(w * scale))
        ch = max(1, int(h * scale))
        self._scale = scale

        # Pre-scale background once — only lightweight overlay redrawn later
        pil_bg = _PILImage.fromarray(image).convert("RGB").resize(
            (cw, ch), _PILImage.LANCZOS
        )
        self._bg_arr = np.array(pil_bg)

        # ── point coordinate text inputs ──────────────────────────────────────
        _ti_kw = dict(style={"description_width": "30px"},
                      layout=widgets.Layout(width="180px"))
        self._x1 = widgets.BoundedIntText(value=w // 4,     min=0, max=w - 1, description="X1", **_ti_kw)
        self._y1 = widgets.BoundedIntText(value=h // 2,     min=0, max=h - 1, description="Y1", **_ti_kw)
        self._x2 = widgets.BoundedIntText(value=3 * w // 4, min=0, max=w - 1, description="X2", **_ti_kw)
        self._y2 = widgets.BoundedIntText(value=h // 2,     min=0, max=h - 1, description="Y2", **_ti_kw)

        # ── measurement controls ──────────────────────────────────────────────
        # min 10 mm (1 cm) — finger reference distance cannot be smaller
        # max 40 mm (4 cm) — a finger is never taller than 4 cm
        self._dist = widgets.BoundedFloatText(
            value=20.0, min=10.0, max=40.0, step=0.5,
            description="Distance (mm):",
            style={"description_width": "120px"},
            layout=widgets.Layout(width="340px"),
        )
        self._dpi_sel = widgets.Dropdown(
            options=[72, 96, 150, 300, 600],
            value=default_dpi,
            description="Print DPI:",
            style={"description_width": "90px"},
            layout=widgets.Layout(width="220px"),
        )
        self._apply_btn = widgets.Button(
            description="Apply Scale", button_style="primary",
            icon="check", layout=widgets.Layout(width="130px"),
        )
        self._reset_btn = widgets.Button(
            description="Reset", button_style="warning",
            icon="refresh", layout=widgets.Layout(width="90px"),
        )

        # ── display widgets ───────────────────────────────────────────────────
        self._px_dist_lbl = widgets.HTML("")
        self._preview_w = widgets.Image(
            value=_render_overlay(self._bg_arr, [], self._scale),
            format="png",
            layout=widgets.Layout(width=f"{cw}px", height=f"{ch}px",
                                  border="1px solid #aaa"),
        )
        self._result = widgets.HTML("")
        # Shown after Apply — scaled image with 10 mm reference grid
        self._grid_lbl = widgets.HTML(
            "<b>Scaled image with 10 mm reference grid:</b>",
            layout=widgets.Layout(display="none", margin="6px 0 2px 0"),
        )
        self._result_img = widgets.Image(
            format="png",
            layout=widgets.Layout(display="none", border="1px solid #888",
                                  margin="0 0 6px 0"),
        )

        # ── wire observers ────────────────────────────────────────────────────
        for sl in (self._x1, self._y1, self._x2, self._y2):
            sl.observe(self._on_coord_change, names="value")
        self._dist.observe(self._on_dist_change, names="value")
        self._dpi_sel.observe(self._on_dpi_change, names="value")
        self._apply_btn.on_click(self._on_apply)
        self._reset_btn.on_click(self._on_reset)

        self._refresh_preview()

        # ── layout ────────────────────────────────────────────────────────────
        _sep = widgets.HTML("<hr style='margin:6px 0'>")
        # Controls column (left) — all inputs + stats table
        _controls = widgets.VBox([
            widgets.HTML(
                "<b>Step 1:</b> Type pixel coordinates into X1/Y1 and X2/Y2 boxes "
                "(press Enter to confirm) — the preview updates and shows the line.<br>"
                "<b>Step 2:</b> Set the real-world distance (10–40 mm) and DPI, "
                "then press <b>Apply Scale</b>."
            ),
            self._px_dist_lbl,
            _sep,
            widgets.HBox([
                widgets.VBox([self._x1, self._y1],
                             layout=widgets.Layout(margin="0 16px 0 0")),
                widgets.VBox([self._x2, self._y2]),
            ]),
            _sep,
            self._dist,
            widgets.HBox([self._dpi_sel, self._apply_btn, self._reset_btn],
                         layout=widgets.Layout(gap="8px", align_items="center",
                                               margin="4px 0")),
            self._result,
            self._grid_lbl,
            self._result_img,
        ], layout=widgets.Layout(min_width="400px"))
        # Preview (first image) on top, controls + grid thumbnail on bottom row
        self._box = widgets.VBox([
            self._preview_w,
            _controls,
        ])

    # ── public ────────────────────────────────────────────────────────────────

    def display(self) -> None:
        display(self._box)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _pts(self) -> list:
        return [(self._x1.value, self._y1.value),
                (self._x2.value, self._y2.value)]

    def _px_dist(self) -> float:
        p1, p2 = self._pts()
        return pixel_distance(p1, p2)

    def _refresh_preview(self) -> None:
        self._preview_w.value = _render_overlay(
            self._bg_arr, self._pts(), self._scale
        )
        d = self._px_dist()
        self._px_dist_lbl.value = f"<b>Pixel distance:</b> {d:.1f} px"

    # ── observers ─────────────────────────────────────────────────────────────

    def _dist_mm(self) -> float:
        """Return the current distance value; returns 0 if blank/NaN."""
        v = self._dist.value
        return v if (v and not (v != v)) else 0.0   # guard NaN

    def _on_coord_change(self, _=None) -> None:
        self._refresh_preview()
        d = self._dist_mm()
        if self.px_per_mm is not None and d > 0:
            self.px_per_mm = self._px_dist() / d
            self._update_result()

    def _on_dist_change(self, _=None) -> None:
        d = self._dist_mm()
        if self.px_per_mm is not None and d > 0:
            self.px_per_mm = self._px_dist() / d
            self._update_result()

    def _on_dpi_change(self, _=None) -> None:
        if self.px_per_mm is not None:
            self._dpi = self._dpi_sel.value
            self._update_result()

    def _on_apply(self, _btn) -> None:
        d = self._px_dist()
        if d < 1:
            self._result.value = "<span style='color:red'>Points are too close together.</span>"
            return
        mm = self._dist_mm()
        if mm <= 0:
            self._result.value = "<span style='color:red'>Enter a valid distance (mm).</span>"
            return
        self.px_per_mm = d / mm
        self._dpi = self._dpi_sel.value
        self._update_result()

    def _on_reset(self, _btn) -> None:
        h, w = self.image.shape[:2]
        self._x1.value, self._y1.value = w // 4,     h // 2
        self._x2.value, self._y2.value = 3 * w // 4, h // 2
        self.px_per_mm = None
        self._result.value = ""
        self._result_img.layout.display = "none"
        self._grid_lbl.layout.display = "none"
        self._refresh_preview()

    def _render_grid(self, scaled_img: np.ndarray, px_per_mm: float) -> tuple:
        """Overlay a 10 mm grid on *scaled_img* and return (png_bytes, w, h)."""
        if scaled_img.ndim == 2:
            pil = _PILImage.fromarray(scaled_img).convert("RGB")
        else:
            pil = _PILImage.fromarray(scaled_img).convert("RGB")
        draw = _ImageDraw.Draw(pil)
        h_px, w_px = scaled_img.shape[:2]
        step = max(1.0, px_per_mm * 10)   # 10 mm in pixels
        x = 0.0
        while x < w_px:
            ix = int(round(x))
            draw.line([(ix, 0), (ix, h_px - 1)], fill=(220, 50, 50), width=1)
            x += step
        y = 0.0
        while y < h_px:
            iy = int(round(y))
            draw.line([(0, iy), (w_px - 1, iy)], fill=(220, 50, 50), width=1)
            y += step
        # Always downscale to the compact grid thumbnail bounds
        disp_scale = min(1.0, _GRID_MAX_W / w_px, _GRID_MAX_H / h_px)
        dw = max(1, int(w_px * disp_scale))
        dh = max(1, int(h_px * disp_scale))
        pil = pil.resize((dw, dh), _PILImage.LANCZOS)
        buf = _io.BytesIO()
        pil.save(buf, format="PNG")
        return buf.getvalue(), dw, dh

    def _update_result(self) -> None:
        target_ppm = self._dpi / 25.4
        scale = target_ppm / self.px_per_mm
        h, w = self.image.shape[:2]
        w_mm, h_mm = w / self.px_per_mm, h / self.px_per_mm
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        sign = "+" if scale >= 1 else ""
        self._result.value = (
            "<table style='border-collapse:collapse;font-size:13px;margin-top:6px'>"
            f"<tr><td style='padding:2px 12px 2px 0'><b>Measured scale</b></td>"
            f"    <td>{self.px_per_mm:.3f} px / mm</td></tr>"
            f"<tr><td><b>Resize factor</b></td>"
            f"    <td>{scale:.4f}×  ({sign}{(scale - 1) * 100:.1f}%)</td></tr>"
            f"<tr><td><b>Physical size of image</b></td>"
            f"    <td>{w_mm:.1f} × {h_mm:.1f} mm</td></tr>"
            f"<tr><td><b>Output @ {self._dpi} DPI</b></td>"
            f"    <td>{new_w} × {new_h} px</td></tr>"
            "</table>"
        )
        # Render scaled image with 10 mm reference grid
        scaled = rescale_to_print_dpi(self.image, self.px_per_mm, self._dpi)
        png_bytes, dw, dh = self._render_grid(scaled, target_ppm)
        self._result_img.value = png_bytes
        self._result_img.layout.width = f"{dw}px"
        self._result_img.layout.height = f"{dh}px"
        self._result_img.layout.display = ""
        self._grid_lbl.layout.display = ""


# ---------------------------------------------------------------------------
# Screen calibration widget
# ---------------------------------------------------------------------------

class ScreenCalibration:
    """
    Calibrate how many CSS pixels equal 1 mm on the physical monitor.

    This is needed so that the pipeline's zoom slider can display the
    fingerprint at true physical size, and so that ``save_with_dpi()``
    can embed the correct DPI for printing.

    Usage
    -----
    >>> sc = ScreenCalibration()
    >>> sc.display()
    >>> # Measure the green bar with a ruler, type measured_mm, press Apply
    >>> pipeline.screen_px_per_mm.value = sc.px_per_mm   # preview only

    IMPORTANT: screen_px_per_mm is used ONLY for the on-screen preview size
    label. It must NOT be used to compute print DPI or physical size.
    Print size must be derived from image_px_per_mm + zoom_factor only.

    How it works
    ------------
    A green bar of ``bar_px`` CSS pixels is drawn.  The user measures it with
    a physical ruler and types the result.  Then::

        screen_px_per_mm = bar_px / measured_mm
    """

    BAR_PX: int = 400      # width of the calibration bar in CSS pixels

    def __init__(self):
        self.px_per_mm: Optional[float] = None

        self._bar = widgets.HTML(
            value=(
                f"<div style='"
                f"width:{self.BAR_PX}px; height:30px; background:#2a9d2a;"
                f"border:1px solid #1a5c1a; box-sizing:border-box;"
                f"'></div>"
                f"<div style='font-size:11px;color:#555;margin-top:2px'>"
                f"← {self.BAR_PX} CSS pixels →  (measure this green bar with a ruler)"
                f"</div>"
            )
        )

        self._mm_input = widgets.BoundedFloatText(
            value=100.0,
            min=1.0,
            max=500.0,
            step=0.5,
            description="Measured (mm):",
            style={"description_width": "110px"},
            layout=widgets.Layout(width="240px"),
        )

        self._apply_btn = widgets.Button(
            description="Apply",
            button_style="primary",
            layout=widgets.Layout(width="80px"),
        )

        self._result = widgets.HTML("")

        self._apply_btn.on_click(self._on_apply)

        self._box = widgets.VBox([
            widgets.HTML(
                "<b>Screen calibration</b><br>"
                "Measure the green bar below with a physical ruler (mm), "
                "enter the value, press <b>Apply</b>."
            ),
            self._bar,
            widgets.HBox([self._mm_input, self._apply_btn]),
            self._result,
        ])

    def display(self) -> None:
        display(self._box)

    def _on_apply(self, _btn) -> None:
        mm = self._mm_input.value
        if mm <= 0:
            self._result.value = "<span style='color:red'>Enter a positive value.</span>"
            return
        self.px_per_mm = self.BAR_PX / mm
        self._result.value = (
            f"<span style='color:green'>"
            f"Screen: <b>{self.px_per_mm:.3f} px/mm</b> &nbsp;|&nbsp; "
            f"{self.px_per_mm * 25.4:.1f} px/inch"
            f"</span><br>"
            f"<small style='color:#888'>For preview only — not used for printing.</small><br>"
            f"<code>pipeline.screen_px_per_mm.value = {self.px_per_mm:.3f}</code>"
        )