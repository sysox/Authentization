from __future__ import annotations

import copy
import inspect
import json
import time
from typing import Any, Callable, Dict, List, Optional, Union

import cv2
import ipywidgets as widgets
import numpy as np

from .utils import normalize_gray


def _array_to_bytes(img: np.ndarray, fmt: str = ".png") -> bytes:
    """Helper to convert numpy array to image bytes for ipywidgets.Image."""
    if img is None:
        return b""

    img_disp = np.asarray(img)

    if img_disp.dtype != np.uint8:
        img_disp = normalize_gray(img_disp)

    if img_disp.ndim == 2:
        pass
    elif img_disp.ndim == 3 and img_disp.shape[2] == 3:
        img_disp = cv2.cvtColor(img_disp, cv2.COLOR_RGB2BGR)
    elif img_disp.ndim == 3 and img_disp.shape[2] == 4:
        img_disp = cv2.cvtColor(img_disp, cv2.COLOR_RGBA2BGRA)
    else:
        return b""

    ok, encoded = cv2.imencode(fmt, img_disp)
    return encoded.tobytes() if ok else b""


def _clone_value(value: Any) -> Any:
    """Clone context value for snapshots."""
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, (dict, list, tuple, set)):
        return copy.deepcopy(value)
    return value


class PipelineContext(dict):
    """
    Shared data container for the pipeline.
    Acts like a dictionary but can have helper methods.
    """

    def __getattr__(self, key: str) -> Any:
        if key in self:
            return self[key]
        return None

    def reset(self, initial_image: np.ndarray):
        self.clear()
        self["original"] = initial_image
        self["image"] = initial_image.copy()
        self["log"] = []

    def snapshot(self) -> Dict[str, Any]:
        return {k: _clone_value(v) for k, v in self.items()}

    def restore(self, snap: Dict[str, Any]):
        self.clear()
        for k, v in snap.items():
            self[k] = _clone_value(v)


class Step:
    """
    Represents a single processing block in the pipeline.
    Supports both:
      - legacy input_keys=['image', 'mask']
      - explicit input_map={'img': 'image', 'mask': 'mask'}
    """

    def __init__(
        self,
        name: str,
        func: Callable,
        input_keys: Optional[List[str]] = None,
        output_keys: Optional[List[str]] = None,
        params: Optional[Dict[str, Union[widgets.Widget, Any]]] = None,
        description: str = "",
        input_map: Optional[Dict[str, str]] = None,
        enabled: bool = True,
        display_key: Optional[str] = None,
        image_scale: Optional[float] = None,
    ):
        self.name = name
        self.func = func
        self.input_keys = input_keys if input_keys else ["image"]
        self.input_map = input_map
        self.output_keys = output_keys if output_keys else ["image"]
        self.params = params if params else {}
        self.description = description
        self.display_key = display_key
        self.image_scale = image_scale  # overrides pipeline-global scale when set

        self.enabled = enabled
        self.last_execution_time = 0.0
        self.index = -1

        # Cache / incremental execution state
        self.is_dirty = True
        self.last_run_ok = False

        # UI Elements
        self.widget_container = None
        self.image_widget = widgets.Image(format="png", width=300, height=300)
        self.controls_container = None
        self.checkbox = widgets.Checkbox(value=enabled, description="Active", indent=False)
        self.status_label = widgets.Label(value="")
        self.description_html = widgets.HTML(
            value=f"<div style='font-size:12px;color:#666'>{self.description}</div>" if self.description else ""
        )

        # Hook up checkbox
        self.checkbox.observe(self._on_enable_change, names="value")

        # Hook up params
        for p in self.params.values():
            if isinstance(p, widgets.Widget):
                p.observe(self._on_param_change, names="value")

        self.parent_pipeline = None

    def _on_enable_change(self, change):
        self.enabled = bool(change["new"])
        self.is_dirty = True
        if self.parent_pipeline:
            self.parent_pipeline.mark_dirty_from(self.index)
            self.parent_pipeline.refresh_layout()
            self.parent_pipeline.run_from(self.index)

    def _on_param_change(self, change):
        self.is_dirty = True
        if self.parent_pipeline:
            self.parent_pipeline.mark_dirty_from(self.index)
            self.parent_pipeline.run_from(self.index)

    def get_ui(
        self,
        layout_mode: str = "fixed",
        image_scale: float = 1.0,
    ) -> Optional[widgets.Widget]:
        effective_scale = self.image_scale if self.image_scale is not None else image_scale
        size = max(50, int(300 * effective_scale))

        if not self.enabled and layout_mode == "dynamic":
            return None

        control_list = [self.checkbox]
        for widget_obj in self.params.values():
            if isinstance(widget_obj, widgets.Widget):
                control_list.append(widget_obj)

        self.controls_container = widgets.VBox(control_list, layout=widgets.Layout(width="100%"))

        if not self.enabled and layout_mode == "fixed":
            placeholder = widgets.HTML(
                value=(
                    f"<div style='width:{size}px; height:{size}px; background:#fff0f0; "
                    "display:flex; align-items:center; justify-content:center; color:#c0392b; "
                    "text-align:center; border:1px dashed #e74c3c;'>"
                    f"Disabled<br>({self.name})</div>"
                )
            )
            return widgets.VBox(
                [
                    widgets.HTML(f"<b style='color:#c0392b'>● {self.name}</b>"),
                    placeholder,
                    self.checkbox,
                    self.description_html,
                    self.status_label,
                ],
                layout=widgets.Layout(border="2px solid #e74c3c", margin="5px", padding="5px", width=f"{size+20}px"),
            )

        box = widgets.VBox(
            [
                widgets.HTML(f"<b style='color:#1a7a1a'>● {self.name}</b>"),
                self.image_widget,
                self.controls_container,
                self.description_html,
                self.status_label,
            ],
            layout=widgets.Layout(border="2px solid #28a745", margin="5px", padding="5px", width=f"{size+20}px"),
        )
        return box

    def _build_kwargs(self, context: PipelineContext) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        missing_inputs: List[str] = []

        if self.input_map:
            for arg_name, ctx_key in self.input_map.items():
                val = context.get(ctx_key)
                if val is None and "mask" not in ctx_key.lower():
                    missing_inputs.append(ctx_key)
                else:
                    kwargs[arg_name] = val
        else:
            sig = inspect.signature(self.func)
            func_params = list(sig.parameters.keys())

            for i, key in enumerate(self.input_keys):
                val = context.get(key)
                if val is None:
                    if "mask" in key.lower():
                        kwargs[key] = None
                    else:
                        missing_inputs.append(key)
                    continue

                if key in func_params:
                    kwargs[key] = val
                elif i < len(func_params):
                    kwargs[func_params[i]] = val
                else:
                    kwargs[key] = val

        for name, widget_obj in self.params.items():
            val = widget_obj.value if isinstance(widget_obj, widgets.Widget) else widget_obj
            kwargs[name] = val

        if missing_inputs:
            raise KeyError(f"Missing inputs: {', '.join(missing_inputs)}")

        return kwargs

    def _update_context_from_result(self, context: PipelineContext, result: Any) -> Any:
        if len(self.output_keys) == 1:
            context[self.output_keys[0]] = result
            return result

        # Handle case where multiple output keys are expected but function returns a single item
        if not isinstance(result, tuple) and len(self.output_keys) > 1:
            # Log a warning or raise an error, then assign to the first output key
            print(f"Warning: Step '{self.name}' expected {len(self.output_keys)} outputs but got a single item. Assigning to '{self.output_keys[0]}'.")
            context[self.output_keys[0]] = result
            return result
        
        if not isinstance(result, tuple) and len(self.output_keys) == 1:
            context[self.output_keys[0]] = result
            return result

        for i, out_key in enumerate(self.output_keys):
            if i < len(result):
                context[out_key] = result[i]
            else:
                # If result tuple is shorter than output_keys, fill remaining with None or warn
                print(f"Warning: Step '{self.name}' result tuple is shorter than output_keys. '{out_key}' not assigned.")
                context[out_key] = None # Assign None to unassigned output keys
        return result[0] if len(result) > 0 else None

    def _resolve_display_data(self, context: PipelineContext, primary_result: Any) -> Any:
        if self.display_key:
            return context.get(self.display_key)
        return primary_result

    def clear_display(self):
        self.image_widget.value = b""
        self.status_label.value = ""

    def run(self, context: PipelineContext) -> bool:
        if not self.enabled:
            self.status_label.value = "Skipped"
            self.last_run_ok = True
            self.is_dirty = False
            return True

        try:
            kwargs = self._build_kwargs(context)
        except Exception as e:
            self.status_label.value = f"Input error: {e}"
            self.last_run_ok = False
            return False

        try:
            t0 = time.time()
            result = self.func(**kwargs)
            dt = time.time() - t0
            self.last_execution_time = dt
            self.status_label.value = f"{dt * 1000:.1f} ms"

            primary_result = self._update_context_from_result(context, result)
            disp_data = self._resolve_display_data(context, primary_result)

            if isinstance(disp_data, np.ndarray):
                self.image_widget.value = _array_to_bytes(disp_data)

            context["log"].append(
                {
                    "step": self.name,
                    "time_ms": round(dt * 1000.0, 2),
                    "enabled": self.enabled,
                }
            )

            self.last_run_ok = True
            self.is_dirty = False
            return True

        except Exception as e:
            self.status_label.value = f"Error: {str(e)}"
            self.last_run_ok = False
            return False


class InteractivePipeline:
    def __init__(self, steps: List[Step], initial_image: np.ndarray = None, preset_name: str = "custom",
                 screen_px_per_mm: float = 3.72):
        self.steps = steps
        self.context = PipelineContext()
        self.initial_image = initial_image
        self.preset_name = preset_name

        for i, step in enumerate(self.steps):
            if step is None:
                raise RuntimeError(f"Pipeline step at index {i} is None")
            step.parent_pipeline = self
            step.index = i

        self.base_snapshot: Optional[Dict[str, Any]] = None
        self.context_snapshots: List[Optional[Dict[str, Any]]] = [None] * len(self.steps)

        self.layout_mode = widgets.ToggleButtons(
            options=["fixed", "dynamic"],
            description="Layout:",
            button_style="",
        )
        self.layout_mode.observe(self._on_layout_change, names="value")

        # UI-only thumbnail zoom — does NOT affect physical size or saving
        self.image_scale = widgets.FloatSlider(
            value=1.0,
            min=0.05,
            max=5.0,
            step=0.05,
            description="UI zoom:",
            continuous_update=False,
            style={"description_width": "60px"},
            layout=widgets.Layout(width="280px"),
        )
        self.image_scale.observe(self._on_scale_change, names="value")

        # Physical size calibration: image pixels per mm (set from ScaleCalibratorIPY).
        # real_mm = (orig_px * zoom_factor) / image_px_per_mm
        self.image_px_per_mm = widgets.BoundedFloatText(
            value=10.0,
            min=0.1,
            max=500.0,
            step=0.01,
            description="img px/mm:",
            tooltip=(
                "Image calibration — pixels per mm in the source image.\n"
                "Measure with ScaleCalibratorIPY (two-point click), then set:\n"
                "  pipeline.image_px_per_mm.value = cal.px_per_mm"
            ),
            style={"description_width": "75px"},
            layout=widgets.Layout(width="175px"),
        )
        self.image_px_per_mm.observe(self._update_size_label_obs, names="value")

        # Physical zoom: user adjusts until on-screen fingerprint matches real finger.
        # real_mm = (orig_px * zoom_factor) / image_px_per_mm
        self.zoom_factor = widgets.FloatSlider(
            value=1.0,
            min=0.1,
            max=10.0,
            step=0.05,
            description="Zoom:",
            continuous_update=True,
            style={"description_width": "45px"},
            layout=widgets.Layout(width="280px"),
        )
        self.zoom_factor.observe(self._update_size_label_obs, names="value")

        # Screen calibration: pixels per mm on the physical monitor — preview only.
        # Use ScreenCalibration widget to measure, then set:
        #   pipeline.screen_px_per_mm.value = sc.px_per_mm
        # This value is ONLY used for the on-screen preview size label.
        # It must NOT influence saving or printing.
        self.screen_px_per_mm = widgets.BoundedFloatText(
            value=screen_px_per_mm,
            min=0.5,
            max=30.0,
            step=0.01,
            description="scr px/mm:",
            tooltip=(
                "Screen calibration — pixels per mm on your monitor (preview only).\n"
                "Use ScreenCalibration widget to measure, then:\n"
                "  pipeline.screen_px_per_mm.value = sc.px_per_mm\n"
                "This does NOT affect saving or printing."
            ),
            style={"description_width": "75px"},
            layout=widgets.Layout(width="175px"),
        )
        self.screen_px_per_mm.observe(self._update_size_label_obs, names="value")

        # Label showing real physical size derived from image calibration + zoom
        self.size_label = widgets.HTML(value="")

        self.rerun_button = widgets.Button(description="Re-run dirty", button_style="info")
        self.rerun_button.on_click(self._on_rerun_button)

        self.run_all_button = widgets.Button(description="Run all", button_style="warning")
        self.run_all_button.on_click(self._on_run_all_button)

        self.grid_box = widgets.HBox([], layout=widgets.Layout(flex_flow="row wrap"))
        self.container = widgets.VBox(
            [
                widgets.HBox([
                    self.layout_mode,
                    self.image_scale,
                ]),
                widgets.HBox([
                    self.image_px_per_mm,
                    self.zoom_factor,
                    self.screen_px_per_mm,
                    self.size_label,
                ]),
                widgets.HBox([self.rerun_button, self.run_all_button]),
                self.grid_box,
            ]
        )

    def _on_layout_change(self, change):
        self.refresh_layout()

    def _on_scale_change(self, change):
        self.refresh_layout()

    def _on_rerun_button(self, btn):
        self.run_from(self.first_dirty_index())

    def _on_run_all_button(self, btn):
        self.mark_dirty_from(0)
        self.run_from(0)

    def refresh_layout(self):
        mode = self.layout_mode.value
        scale = self.image_scale.value
        children = []
        for step in self.steps:
            ui = step.get_ui(layout_mode=mode, image_scale=scale)
            if ui is not None:
                children.append(ui)
        self.grid_box.children = children

    def first_dirty_index(self) -> int:
        for i, step in enumerate(self.steps):
            if step.is_dirty:
                return i
        return len(self.steps)

    def mark_dirty_from(self, start_index: int):
        if start_index < 0:
            start_index = 0
        for i in range(start_index, len(self.steps)):
            self.steps[i].is_dirty = True
            self.context_snapshots[i] = None

    def _ensure_base_snapshot(self):
        if self.initial_image is None:
            return
        self.context.reset(self.initial_image)
        self.base_snapshot = self.context.snapshot()

    def _restore_for_start(self, start_index: int):
        if self.initial_image is None:
            return

        if self.base_snapshot is None:
            self._ensure_base_snapshot()

        if start_index <= 0:
            self.context.restore(self.base_snapshot)
            return

        prev_snap = self.context_snapshots[start_index - 1]
        if prev_snap is None:
            self.run_from(start_index - 1)
            prev_snap = self.context_snapshots[start_index - 1]

        if prev_snap is None:
            self.context.restore(self.base_snapshot)
        else:
            self.context.restore(prev_snap)

    def run_from(self, start_index: int = 0):
        if self.initial_image is None:
            return

        if start_index >= len(self.steps):
            return

        if start_index < 0:
            start_index = 0

        self._restore_for_start(start_index)

        for i in range(start_index, len(self.steps)):
            step = self.steps[i]
            ok = step.run(self.context)
            self.context_snapshots[i] = self.context.snapshot() if ok else None
            if not ok:
                for j in range(i + 1, len(self.steps)):
                    self.steps[j].status_label.value = "Blocked by previous error"
                break

    def run(self):
        self.mark_dirty_from(0)
        self.run_from(0)

    def display(self):
        self.refresh_layout()
        self.run()
        return self.container

    def set_image(self, img: np.ndarray):
        self.initial_image = img
        self.base_snapshot = None
        self.context_snapshots = [None] * len(self.steps)
        self.mark_dirty_from(0)
        self.run_from(0)

    def _update_size_label_obs(self, _change=None):
        """Recompute and show real physical size when zoom/calibration changes."""
        if self.initial_image is None:
            return
        orig_w = self.initial_image.shape[1]
        orig_h = self.initial_image.shape[0]
        img_px_per_mm = self.image_px_per_mm.value
        zoom = self.zoom_factor.value
        scr_px_per_mm = self.screen_px_per_mm.value
        if img_px_per_mm <= 0:
            self.size_label.value = ""
            return
        # canonical physical size — single source of truth
        real_w_mm = (orig_w * zoom) / img_px_per_mm
        real_h_mm = (orig_h * zoom) / img_px_per_mm
        # screen preview size (informational only)
        disp_w_px = real_w_mm * scr_px_per_mm
        self.size_label.value = (
            f"<span style='font-size:12px;color:#555;padding:4px'>"
            f"real: <b>{real_w_mm:.1f}&times;{real_h_mm:.1f} mm</b>"
            f" &nbsp;|&nbsp; screen: ~{disp_w_px:.0f} px wide</span>"
        )

    def save_with_dpi(self, output_path: str, context_key: str = "binary") -> float:
        """
        Save a processed image with DPI embedded so that printing at 100% scale
        reproduces the correct physical size.

        Physical size is derived exclusively from image calibration and zoom:
            real_mm  = (original_px * zoom_factor) / image_px_per_mm
            image_dpi = saved_px / (real_mm / 25.4)

        screen_px_per_mm is NOT used here — it is for preview only.

        Parameters
        ----------
        output_path : str
            Destination file path (.png recommended).
        context_key : str
            Key in the pipeline context to save (default: 'binary').

        Returns
        -------
        float
            The DPI value that was embedded.
        """
        from PIL import Image as PILImage

        img_arr = self.context.get(context_key)
        if img_arr is None:
            raise KeyError(f"Context key '{context_key}' not found. Run the pipeline first.")
        if not isinstance(img_arr, np.ndarray):
            raise TypeError(f"Context['{context_key}'] is not a numpy array.")

        img_px_per_mm = self.image_px_per_mm.value
        if img_px_per_mm <= 0:
            raise ValueError("image_px_per_mm must be > 0. Run ScaleCalibratorIPY first.")

        zoom = self.zoom_factor.value
        orig_w = self.initial_image.shape[1]
        saved_w = img_arr.shape[1]

        # canonical physical size — no screen dependency
        real_mm = (orig_w * zoom) / img_px_per_mm
        image_dpi = saved_w / (real_mm / 25.4)

        arr = img_arr
        if arr.dtype != np.uint8:
            from .utils import normalize_gray
            arr = normalize_gray(arr)

        if arr.ndim == 2:
            pil_img = PILImage.fromarray(arr, mode="L")
        elif arr.ndim == 3 and arr.shape[2] == 3:
            pil_img = PILImage.fromarray(arr, mode="RGB")
        else:
            pil_img = PILImage.fromarray(arr)

        pil_img.save(output_path, dpi=(image_dpi, image_dpi))

        print(f"[INFO] Saved:          {output_path}")
        print(f"[INFO] real_mm:        {real_mm:.2f} mm  (= orig_px × zoom / img_px_per_mm)")
        print(f"[INFO] DPI embedded:   {image_dpi:.2f}")
        print(f"[INFO] Physical width: {real_mm:.1f} mm  ({real_mm/10:.1f} cm)")
        return image_dpi

    def get_params(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "preset_name": self.preset_name,
            "steps": {},
        }
        for step in self.steps:
            step_data = {
                "enabled": step.enabled,
                "params": {},
            }
            for name, widget_obj in step.params.items():
                value = widget_obj.value if isinstance(widget_obj, widgets.Widget) else widget_obj
                step_data["params"][name] = value
            data["steps"][step.name] = step_data
        return data

    def set_params(self, config: Dict[str, Any], rerun: bool = True):
        step_cfg = config.get("steps", {})
        for step in self.steps:
            cfg = step_cfg.get(step.name)
            if not cfg:
                continue

            enabled = cfg.get("enabled", step.enabled)
            step.enabled = bool(enabled)
            step.checkbox.value = bool(enabled)

            params_cfg = cfg.get("params", {})
            for name, value in params_cfg.items():
                if name not in step.params:
                    continue
                widget_obj = step.params[name]
                if isinstance(widget_obj, widgets.Widget):
                    try:
                        widget_obj.value = value
                    except Exception:
                        pass
                else:
                    step.params[name] = value

        self.refresh_layout()
        self.mark_dirty_from(0)
        if rerun:
            self.run_from(0)

    def export_params_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.get_params(), f, indent=2)

    def import_params_json(self, path: str, rerun: bool = True):
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)
        self.set_params(config, rerun=rerun)
