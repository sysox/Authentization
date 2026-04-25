from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple

import ipywidgets as widgets
import numpy as np
import cv2

from .interactive import InteractivePipeline, Step


# ============================================================
# OPTIONAL IMPORTS
# ============================================================

def _optional_attr(module_name: str, attr_name: str):
    try:
        module = __import__(f"{__package__}.{module_name}", fromlist=[attr_name])
        return getattr(module, attr_name, None)
    except Exception:
        return None

# Gray
compute_gray_variants = _optional_attr("gray", "compute_gray_variants")

# Enhancement / preprocessing
apply_clahe = _optional_attr("enhancement", "apply_clahe")
median_blur = _optional_attr("enhancement", "median_blur")
difference_of_gaussians = _optional_attr("enhancement", "difference_of_gaussians")
local_contrast_enhancement = _optional_attr("enhancement", "local_contrast_enhancement")
contrast_stretch_saturated = _optional_attr("enhancement", "contrast_stretch_saturated")
gamma_correction = _optional_attr("enhancement", "gamma_correction")
dynamic_gabor_pipeline = _optional_attr("enhancement", "dynamic_gabor_pipeline")
apply_ridge_filter = _optional_attr("enhancement", "apply_ridge_filter")
fft_gaussian_bandpass = _optional_attr("enhancement", "fft_gaussian_bandpass")
coherence_enhancing_diffusion = _optional_attr("enhancement", "coherence_enhancing_diffusion")
gabor_rebuild = _optional_attr("enhancement", "gabor_rebuild")
gaussian_blur_masked = _optional_attr("filters", "gaussian_blur_masked")

# Mask / ROI
build_foreground_mask = _optional_attr("mask", "build_foreground_mask")
mask_from_black_background = _optional_attr("mask", "mask_from_black_background")
apply_mask_outside_zero = _optional_attr("mask", "apply_mask_outside_zero")
compute_fingerprint_roi = _optional_attr("mask", "compute_fingerprint_roi")

# Binarization / morphology / skeleton / minutiae
binarize_image = _optional_attr("binarization", "binarize_image")
apply_mask_to_binary = _optional_attr("binarization", "apply_mask_to_binary")
apply_morphology = _optional_attr("morphology", "apply_morphology")
get_skeleton = _optional_attr("skeleton", "get_skeleton")
prune_skeleton_topology = _optional_attr("skeleton", "prune_skeleton_topology")
extract_minutiae_crossing_number = _optional_attr("minutiae", "extract_minutiae_crossing_number")
finalize_minutiae = _optional_attr("minutiae", "finalize_minutiae")
draw_minutiae = _optional_attr("minutiae", "draw_minutiae")


# ============================================================
# HELPERS
# ============================================================

@dataclass
class OperationSpec:
    key: str
    name: str
    func: Callable
    input_map: Dict[str, str]
    output_keys: List[str]
    params_spec: Dict[str, Dict[str, Any]]
    description: str = ""
    display_key: Optional[str] = None
    level: str = "core"


def _safe_signature_params(func: Callable) -> set[str]:
    try:
        return set(inspect.signature(func).parameters.keys())
    except Exception:
        return set()


def _pick_input_map(func: Callable, candidates: Dict[str, str]) -> Dict[str, str]:
    """
    candidates: {param_name: context_key}
    Returns entries where param_name is in func's signature.
    """
    params = _safe_signature_params(func)
    return {param: ctx_key for param, ctx_key in candidates.items() if param in params}


def _slider_from_spec(name: str, spec: Dict[str, Any]):
    kind = spec.get("kind", "int")
    label = spec.get("label", name)

    if kind == "int":
        return widgets.IntSlider(
            value=spec["value"],
            min=spec["min"],
            max=spec["max"],
            step=spec.get("step", 1),
            description=label,
            continuous_update=spec.get("continuous_update", False),
        )

    if kind == "float":
        return widgets.FloatSlider(
            value=spec["value"],
            min=spec["min"],
            max=spec["max"],
            step=spec.get("step", 0.1),
            description=label,
            continuous_update=spec.get("continuous_update", False),
        )

    if kind == "bool":
        return widgets.Checkbox(
            value=spec["value"],
            description=label,
            indent=False,
        )

    if kind == "choice":
        return widgets.Dropdown(
            options=spec["options"],
            value=spec["value"],
            description=label,
        )
    
    if kind == "fixed":
        return spec["value"] 

    raise ValueError(f"Unknown widget kind: {kind}")


def _build_params_widgets(params_spec: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name, spec in params_spec.items():
        if spec.get("kind") == "fixed":
            out[name] = spec["value"]
        else:
            out[name] = _slider_from_spec(name, spec)
    return out


def _filter_params_to_signature(func, params_spec: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    try:
        sig = inspect.signature(func)
        allowed = set(sig.parameters.keys())
        
        # If func accepts **kwargs, allow all params
        for param in sig.parameters.values():
            if param.kind == inspect.Parameter.VAR_KEYWORD:
                return params_spec
                
    except Exception:
        return params_spec

    out = {}
    for name, spec in params_spec.items():
        if name in allowed:
            out[name] = spec
    return out


def _merge_param_specs(
    base: Dict[str, Dict[str, Any]],
    overrides: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    out = {k: dict(v) for k, v in base.items()}
    if overrides:
        for key, value in overrides.items():
            out[key] = dict(value)
    return out


def _make_step(spec: OperationSpec, enabled: bool = True, override_params: Optional[Dict[str, Dict[str, Any]]] = None) -> Step:
    params_spec = dict(spec.params_spec)
    if override_params:
        for key, value in override_params.items():
            params_spec[key] = value

    params_spec = _filter_params_to_signature(spec.func, params_spec)
    return Step(
        name=spec.name,
        func=spec.func,
        input_map=spec.input_map,
        output_keys=spec.output_keys,
        params=_build_params_widgets(params_spec),
        description=spec.description,
        enabled=enabled,
        display_key=spec.display_key,
    )

# ============================================================
# Visualization Helpers
# ============================================================

def visualize_skeleton_overlay(image: np.ndarray, skel: np.ndarray) -> np.ndarray:
    """Helper to draw skeleton (green) on top of original image."""
    if image is None:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Convert to RGB
    if len(image.shape) == 2:
        out = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        out = image.copy()
        
    if skel is not None:
        # Draw skeleton in green
        mask_sk = skel > 0
        out[mask_sk] = [0, 255, 0]
    return out

def visualize_minutiae_overlay(image: np.ndarray, minutiae_list: list) -> np.ndarray:
    """Draw minutiae on *image*. Delegates to :func:`fingerprints.minutiae.draw_minutiae`."""
    if image is None:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    return draw_minutiae(minutiae_list, background=image)

# ============================================================
# Step Wrappers
# ============================================================

def step_mask_vis_roi(image: np.ndarray, **kwargs):
    if compute_fingerprint_roi is None:
        raise RuntimeError("compute_fingerprint_roi is not available.")
    
    mask_img = compute_fingerprint_roi(image, **kwargs)
    
    vis = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    vis[mask_img == 0] = vis[mask_img == 0] // 2 + [128, 0, 0]
    
    out = image.copy()
    out[mask_img == 0] = 255 
    return vis, out, mask_img

def step_preserve_original(image: np.ndarray) -> np.ndarray:
    """Store the raw image as 'original_image' in context before any masking/processing."""
    return image


def step_mask_black_bg_vis(image: np.ndarray, thresh: int = 31, open_k: int = 7, close_k: int = 9, min_area: int = 0):
    """Threshold-based ROI mask (dark background). Returns overlay vis, masked image, mask."""
    if mask_from_black_background is None:
        raise RuntimeError("mask_from_black_background is not available.")
    mask = mask_from_black_background(image, thresh=thresh, open_k=open_k, close_k=close_k, min_area=min_area)

    # Overlay: dim + tint pixels outside mask so user sees the boundary
    vis = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    outside = mask == 0
    vis[outside] = (vis[outside].astype(np.float32) * 0.35 + np.array([80, 0, 0])).clip(0, 255).astype(np.uint8)

    # Masked grayscale image (zeros outside)
    masked = image.copy()
    masked[outside] = 0

    return vis, masked, mask


def step_display_context(img: np.ndarray) -> np.ndarray:
    """Identity passthrough — used to display an intermediate context value as its own step panel."""
    return img


def step_skeleton_only_vis(image: np.ndarray, **kwargs):
    """Skeletonize binary image and return the skeleton as white-on-black (no background overlay)."""
    if get_skeleton is None:
        raise RuntimeError("get_skeleton is not available.")
    skel = get_skeleton(image, **kwargs)
    skel_uint8 = (skel > 0).astype(np.uint8) * 255
    return skel_uint8, skel


def step_prune_skeleton_only_vis(skeleton: np.ndarray, **kwargs):
    """Prune skeleton and return pruned skeleton as white-on-black (no background overlay)."""
    if prune_skeleton_topology is None:
        raise RuntimeError("prune_skeleton_topology is not available.")
    pruned = prune_skeleton_topology(skeleton, **kwargs)
    pruned_uint8 = (pruned > 0).astype(np.uint8) * 255
    return pruned_uint8, pruned


def step_minutiae_on_original(original_image: np.ndarray, minutiae_list: list) -> np.ndarray:
    """Draw minutiae on the original (pre-masking) grayscale image."""
    return visualize_minutiae_overlay(original_image, minutiae_list)


def step_skeletonize_vis(image: np.ndarray, **kwargs):
    if get_skeleton is None:
        raise RuntimeError("get_skeleton is not available.")
    skel = get_skeleton(image, **kwargs)
    vis = visualize_skeleton_overlay(image, skel)
    return vis, skel

def step_minutiae_vis(
    skeleton_img: np.ndarray,
    mask: Optional[np.ndarray] = None,
    border_margin: int = 10,
    cluster_dist: int = 6,
    swap_on_inverted_polarity: bool = True,
    use_traced_orientation: bool = True,
    max_spur_len: int = 15,
    suppress_dist: int = 12,
):
    if extract_minutiae_crossing_number is None or finalize_minutiae is None:
        raise RuntimeError("Minutiae functions are not available.")
        
    # Extraction
    min_list = extract_minutiae_crossing_number(
        skeleton_img, 
        mask=mask,
        border_margin=border_margin,
        cluster_dist=cluster_dist,
        swap_on_inverted_polarity=swap_on_inverted_polarity,
        use_traced_orientation=use_traced_orientation
    )
    
    # Cleaning
    min_list = finalize_minutiae(
        min_list, 
        skeleton_img,
        max_spur_len=max_spur_len,
        suppress_dist=suppress_dist
    )
    
    vis = visualize_minutiae_overlay(skeleton_img, min_list)
    return vis, min_list

def step_channel_select(img: np.ndarray, channel: str):
    if compute_gray_variants is None:
        raise RuntimeError("compute_gray_variants is not available.")
    variants = compute_gray_variants(img)
    return variants.get(channel, variants['gray_default'])

def step_invert(img: np.ndarray, active: bool):
    return cv2.bitwise_not(img) if active else img

def step_prune_skeleton_vis(skeleton: np.ndarray, binary_image: np.ndarray, **kwargs):
    """Prunes skeleton and returns a visualization using the binary image as background."""
    if prune_skeleton_topology is None:
        raise RuntimeError("prune_skeleton_topology is not available.")
    
    updated_skeleton = prune_skeleton_topology(skeleton, **kwargs)
    vis = visualize_skeleton_overlay(binary_image, updated_skeleton)
    return vis, updated_skeleton

# ============================================================
# AVAILABLE OPERATIONS
# ============================================================

def _available_operation_specs() -> Dict[str, OperationSpec]:
    ops: Dict[str, OperationSpec] = {}

    # --- Channel Selection ---
    if compute_gray_variants is not None:
         ops["channel_select"] = OperationSpec(
            key="channel_select",
            name="Channel Selection",
            func=step_channel_select,
            input_map={"img": "original"}, # Takes original image from context
            output_keys=["image"], # Updates main processing image
            params_spec={
                "channel": {"kind": "choice", "value": "gray_green", "options": ["gray_luma", "gray_green", "gray_blue", "gray_red", "gray_max"], "label": "Channel"},
            },
            description="Select best color channel from RGB input.",
            display_key="image",
            level="core",
        )

    # --- Mask / ROI ---
    if mask_from_black_background is not None:
        ops["mask_black_bg"] = OperationSpec(
            key="mask_black_bg",
            name="Mask from Black Background",
            func=mask_from_black_background,
            input_map=_pick_input_map(
                mask_from_black_background,
                {"image": ["gray", "img_gray", "img"]},
            ),
            output_keys=["mask"],
            params_spec={
                "thresh": {"kind": "int", "value": 20, "min": 0, "max": 80, "step": 1, "label": "Thresh"},
                "invert": {"kind": "bool", "value": False, "label": "Invert"},
                "open_k": {"kind": "int", "value": 5, "min": 0, "max": 21, "step": 2, "label": "Open K"},
                "close_k": {"kind": "int", "value": 9, "min": 0, "max": 21, "step": 2, "label": "Close K"},
                "min_area": {"kind": "int", "value": 1000, "min": 0, "max": 10000, "step": 100, "label": "Min Area"},
            },
            description="Estimate ROI mask from dark background around the fingerprint.",
            display_key="mask",
            level="core",
        )

    if build_foreground_mask is not None:
        ops["foreground_mask"] = OperationSpec(
            key="foreground_mask",
            name="Foreground Mask",
            func=build_foreground_mask,
            input_map={"gray": "image"},
            output_keys=["mask"],
            params_spec={
                "blur_k": {"kind": "int", "value": 9, "min": 1, "max": 21, "step": 2, "label": "Blur"},
                "block": {"kind": "int", "value": 31, "min": 3, "max": 81, "step": 2, "label": "Block"},
                "C": {"kind": "int", "value": 7, "min": -10, "max": 20, "step": 1, "label": "C"},
                "close_k": {"kind": "int", "value": 9, "min": 0, "max": 21, "step": 2, "label": "Close"},
                "open_k": {"kind": "int", "value": 5, "min": 0, "max": 21, "step": 2, "label": "Open"},
                "min_area": {"kind": "int", "value": 4000, "min": 0, "max": 20000, "step": 100, "label": "Min area"},
            },
            description="Find foreground region containing usable ridge information.",
            display_key="mask",
        )

    ops["roi_segmentation_vis"] = OperationSpec(
        key="roi_segmentation_vis",
        name="ROI Segmentation",
        func=step_mask_vis_roi,
        input_map={"image": "image"},
        output_keys=["roi_vis", "image", "mask"], 
        params_spec={
            "block": {"kind": "int", "value": 16, "min": 8, "max": 64, "step": 4, "label": "Block Size"},
            "close_k": {"kind": "int", "value": 25, "min": 1, "max": 51, "step": 2, "label": "Close K"},
            "erode_k": {"kind": "int", "value": 7, "min": 1, "max": 21, "step": 2, "label": "Erode K"},
            "close_iter": {"kind": "int", "value": 4, "min": 1, "max": 10, "step": 1, "label": "Close Iter"},
            "dilate_iter": {"kind": "int", "value": 1, "min": 0, "max": 5, "step": 1, "label": "Dilate Iter"},
            "min_area": {"kind": "int", "value": 4000, "min": 100, "max": 20000, "step": 100, "label": "Min Area"},
        },
        description="Variance-based ROI extraction with visualization.",
        display_key="roi_vis",
        level="core",
    )

    if apply_mask_outside_zero is not None:
        ops["apply_mask_zero"] = OperationSpec(
            key="apply_mask_zero",
            name="Apply Mask (Zero)",
            func=apply_mask_outside_zero,
            input_map=_pick_input_map(
                apply_mask_outside_zero,
                {
                    "img": "image",
                    "mask": "mask",
                },
            ),
            output_keys=["image"],
            params_spec={},
            description="Zero pixels outside mask.",
            level="core",
        )
    
    ops["preserve_original"] = OperationSpec(
        key="preserve_original",
        name="Original Image",
        func=step_preserve_original,
        input_map={"image": "image"},
        output_keys=["original_image"],
        params_spec={},
        description="Display original image and store it for later overlays.",
        display_key="original_image",
        level="core",
    )

    if mask_from_black_background is not None:
        _mask_bg_params = {
            "thresh":   {"kind": "int", "value": 31, "min": 0,  "max": 100,   "step": 1,   "label": "BG Thresh"},
            "open_k":   {"kind": "int", "value": 7,  "min": 1,  "max": 31,    "step": 2,   "label": "Open K"},
            "close_k":  {"kind": "int", "value": 9,  "min": 1,  "max": 41,    "step": 2,   "label": "Close K"},
            "min_area": {"kind": "int", "value": 0,  "min": 0,  "max": 10000, "step": 100, "label": "Min Area"},
        }
        ops["mask_black_bg_vis"] = OperationSpec(
            key="mask_black_bg_vis",
            name="ROI Mask (overlay)",
            func=step_mask_black_bg_vis,
            input_map={"image": "image"},
            output_keys=["roi_vis", "image", "mask"],
            params_spec=_mask_bg_params,
            description="Threshold-based ROI mask — shows original with dimmed outside area.",
            display_key="roi_vis",
            level="core",
        )
        ops["mask_black_bg_thresh"] = OperationSpec(
            key="mask_black_bg_thresh",
            name="ROI Mask",
            func=step_mask_black_bg_vis,
            input_map={"image": "image"},
            output_keys=["roi_vis", "image", "mask"],
            params_spec=_mask_bg_params,
            description="Threshold-based ROI mask — shows computed mask.",
            display_key="mask",
            level="core",
        )

    ops["display_mask"] = OperationSpec(
        key="display_mask",
        name="Show Mask",
        func=step_display_context,
        input_map={"img": "mask"},
        output_keys=["mask"],
        params_spec={},
        description="Display the current mask.",
        display_key="mask",
        level="core",
    )

    ops["display_image"] = OperationSpec(
        key="display_image",
        name="Show Image",
        func=step_display_context,
        input_map={"img": "image"},
        output_keys=["image"],
        params_spec={},
        description="Display the current image.",
        display_key="image",
        level="core",
    )

    ops["minutiae_on_original"] = OperationSpec(
        key="minutiae_on_original",
        name="Minutiae on Original",
        func=step_minutiae_on_original,
        input_map={"original_image": "original_image", "minutiae_list": "minutiae_list"},
        output_keys=["minutiae_on_orig"],
        params_spec={},
        description="Draw minutiae on the original (pre-masking) image.",
        display_key="minutiae_on_orig",
        level="core",
    )

    # --- Enhancement ---
    if contrast_stretch_saturated is not None:
        ops["stretch"] = OperationSpec(
            key="stretch",
            name="Contrast Stretch",
            func=contrast_stretch_saturated,
            input_map=_pick_input_map(
                contrast_stretch_saturated,
                {"img_gray": "image"},
            ),
            output_keys=["image"],
            params_spec={
                "saturated": {"kind": "float", "value": 0.35, "min": 0.0, "max": 2.0, "step": 0.05, "label": "Sat %"},
            },
            description="Simple contrast rescaling using saturated tails.",
            level="advanced",
        )

    if gamma_correction is not None:
        ops["gamma"] = OperationSpec(
            key="gamma",
            name="Gamma Correction",
            func=gamma_correction,
            input_map=_pick_input_map(
                gamma_correction,
                {"img_gray": "image"},
            ),
            output_keys=["image"],
            params_spec={
                "gamma": {"kind": "float", "value": 1.0, "min": 0.2, "max": 3.0, "step": 0.05, "label": "Gamma"},
            },
            description="Adjust global brightness / ridge contrast.",
            level="advanced",
        )

    if apply_clahe is not None:
        ops["clahe"] = OperationSpec(
            key="clahe",
            name="CLAHE",
            func=apply_clahe,
            input_map=_pick_input_map(
                apply_clahe,
                {"img_gray": "image"},
            ),
            output_keys=["image"],
            params_spec={
                "clip": {"kind": "float", "value": 2.5, "min": 1.0, "max": 5.0, "step": 0.1, "label": "Clip"},
                "tile": {"kind": "int", "value": 8, "min": 4, "max": 16, "step": 2, "label": "Tile"},
            },
            description="Local histogram equalization.",
            level="core",
        )

    if median_blur is not None:
        ops["median_blur"] = OperationSpec(
            key="median_blur",
            name="Median Blur",
            func=median_blur,
            input_map=_pick_input_map(median_blur, {"img_gray": "image"}),
            output_keys=["image"],
            params_spec={
                "ksize": {"kind": "int", "value": 5, "min": 1, "max": 21, "step": 2, "label": "Kernel"},
            },
            description="Median blur — removes salt-and-pepper noise.",
            display_key="image",
            level="advanced",
        )

    if gaussian_blur_masked is not None:
        ops["gaussian_blur"] = OperationSpec(
            key="gaussian_blur",
            name="Gaussian Blur",
            func=gaussian_blur_masked,
            input_map=_pick_input_map(gaussian_blur_masked, {"img": "image"}),
            output_keys=["image"],
            params_spec={
                "ksize": {"kind": "int", "value": 7, "min": 1, "max": 21, "step": 2, "label": "Kernel"},
            },
            description="Gaussian blur — smooths ridges before binarisation.",
            display_key="image",
            level="advanced",
        )

    if local_contrast_enhancement is not None:
        ops["local_contrast"] = OperationSpec(
            key="local_contrast",
            name="Local Contrast",
            func=local_contrast_enhancement,
            input_map=_pick_input_map(
                local_contrast_enhancement,
                {"img_gray": "image"},
            ),
            output_keys=["image"],
            params_spec={
                "radius": {"kind": "int", "value": 31, "min": 3, "max": 81, "step": 2, "label": "Radius"},
                "amount": {"kind": "float", "value": 1.0, "min": 0.2, "max": 3.0, "step": 0.1, "label": "Amount"},
            },
            description="Local contrast boosting for uneven illumination.",
            level="advanced",
        )

    if difference_of_gaussians is not None:
        ops["dog"] = OperationSpec(
            key="dog",
            name="Difference of Gaussians",
            func=difference_of_gaussians,
            input_map=_pick_input_map(
                difference_of_gaussians,
                {"img": "image"},
            ),
            output_keys=["image"],
            params_spec={
                "sigma1": {"kind": "float", "value": 1.0, "min": 0.2, "max": 8.0, "step": 0.1, "label": "Sigma 1"},
                "sigma2": {"kind": "float", "value": 8.0, "min": 0.5, "max": 20.0, "step": 0.1, "label": "Sigma 2"},
                "gain": {"kind": "float", "value": 1.5, "min": 0.2, "max": 4.0, "step": 0.1, "label": "Gain"},
            },
            description="Band-pass like enhancement.",
            level="advanced",
        )

    if fft_gaussian_bandpass is not None:
        ops["fft_bandpass"] = OperationSpec(
            key="fft_bandpass",
            name="FFT Bandpass",
            func=fft_gaussian_bandpass,
            input_map=_pick_input_map(
                fft_gaussian_bandpass,
                {"img": "image"},
            ),
            output_keys=["image"],
            params_spec={
                "low_sigma": {"kind": "float", "value": 3.0, "min": 0.5, "max": 30.0, "step": 0.5, "label": "Low σ"},
                "high_sigma": {"kind": "float", "value": 28.0, "min": 2.0, "max": 100.0, "step": 1.0, "label": "High σ"},
            },
            description="Frequency-domain band-pass enhancement.",
            level="experimental",
        )

    if coherence_enhancing_diffusion is not None:
        ops["ced"] = OperationSpec(
            key="ced",
            name="CED",
            func=coherence_enhancing_diffusion,
            input_map=_pick_input_map(
                coherence_enhancing_diffusion,
                {"img": "image"},
            ),
            output_keys=["image"],
            params_spec={
                "lambda_": {"kind": "float", "value": 0.1, "min": 0.01, "max": 1.0, "step": 0.01, "label": "Lambda"},
                "sigma": {"kind": "float", "value": 1.0, "min": 0.1, "max": 5.0, "step": 0.1, "label": "Sigma"},
                "rho": {"kind": "float", "value": 3.0, "min": 1.0, "max": 10.0, "step": 0.1, "label": "Rho"},
                "step_size": {"kind": "float", "value": 0.2, "min": 0.05, "max": 0.5, "step": 0.01, "label": "Step"},
                "m": {"kind": "float", "value": 1.0, "min": 0.5, "max": 2.0, "step": 0.1, "label": "M"},
                "n_steps": {"kind": "int", "value": 5, "min": 1, "max": 20, "step": 1, "label": "Steps"},
            },
            description="Coherence-enhancing diffusion.",
            level="experimental",
        )

    if apply_ridge_filter is not None:
        ops["ridge_filter"] = OperationSpec(
            key="ridge_filter",
            name="Ridge Filter",
            func=apply_ridge_filter,
            input_map=_pick_input_map(
                apply_ridge_filter,
                {"img_gray": "image"},
            ),
            output_keys=["image"],
            params_spec={
                "method": {"kind": "choice", "value": "meijering", "options": ["meijering", "frangi", "sato", "hessian"], "label": "Method"},
                "sigmas": {"kind": "fixed", "value": (1.0, 2.0, 3.0)}, # Fixed tuple
                "black_ridges": {"kind": "bool", "value": True, "label": "Black Ridges"},
            },
            description="Ridge-like structure enhancement.",
            level="advanced",
        )

    if dynamic_gabor_pipeline is not None:
        ops["dynamic_gabor"] = OperationSpec(
            key="dynamic_gabor",
            name="Dynamic Gabor",
            func=dynamic_gabor_pipeline,
            input_map=_pick_input_map(
                dynamic_gabor_pipeline,
                {"gray": "image"},
            ),
            output_keys=["image", "orientation", "frequency"],
            params_spec={
                "orient_block": {"kind": "int", "value": 16, "min": 4, "max": 64, "step": 2, "label": "Ori blk"},
                "orient_smooth": {"kind": "int", "value": 5, "min": 1, "max": 21, "step": 2, "label": "Ori sm"},
                "freq_block": {"kind": "int", "value": 32, "min": 8, "max": 96, "step": 2, "label": "Freq blk"},
                "min_wave": {"kind": "int", "value": 4, "min": 2, "max": 20, "step": 1, "label": "Min wave"},
                "max_wave": {"kind": "int", "value": 20, "min": 4, "max": 40, "step": 1, "label": "Max wave"},
                "default_freq": {"kind": "float", "value": 0.08, "min": 0.02, "max": 0.25, "step": 0.01, "label": "Def freq"},
                "sigma_base": {"kind": "float", "value": 4.0, "min": 1.0, "max": 12.0, "step": 0.1, "label": "Sigma"},
                "gamma": {"kind": "float", "value": 0.5, "min": 0.1, "max": 2.0, "step": 0.05, "label": "Gamma"},
                "kernel_size": {"kind": "int", "value": 21, "min": 7, "max": 51, "step": 2, "label": "Kernel"},
            },
            description="Estimate orientation/frequency and apply contextual Gabor enhancement.",
            display_key="image",
            level="experimental",
        )

    if gabor_rebuild is not None:
        ops["gabor_rebuild"] = OperationSpec(
            key="gabor_rebuild",
            name="Gabor Rebuild",
            func=gabor_rebuild,
            input_map=_pick_input_map(
                gabor_rebuild,
                {
                    "img_gray": "image",
                    "theta_map": "orientation",
                    "freq_map": "frequency",
                },
            ),
            output_keys=["image"],
            params_spec={
                "sigma": {"kind": "float", "value": 3.0, "min": 1.0, "max": 12.0, "step": 0.1, "label": "Sigma"},
                "freq": {"kind": "float", "value": 0.1, "min": 0.01, "max": 0.5, "step": 0.01, "label": "Freq"},
                "kernel_size": {"kind": "int", "value": 21, "min": 7, "max": 51, "step": 2, "label": "Kernel"},
                "gamma": {"kind": "float", "value": 0.5, "min": 0.1, "max": 2.0, "step": 0.05, "label": "Gamma"},
                "n_angles": {"kind": "int", "value": 16, "min": 4, "max": 32, "step": 2, "label": "Angles"},
            },
            description="Rebuild enhanced image from precomputed orientation/frequency maps.",
            level="experimental",
        )

    # --- Binarization ---
    if binarize_image is not None:
        ops["binarize"] = OperationSpec(
            key="binarize",
            name="Binarization",
            func=binarize_image,
            input_map=_pick_input_map(
                binarize_image,
                {"img_gray": "image"},
            ),
            output_keys=["binary"],
            params_spec={
                "method": {"kind": "choice", "value": "adaptive", "options": ["adaptive", "otsu", "sauvola", "niblack", "fixed"], "label": "Method"},
                "block": {"kind": "int", "value": 31, "min": 9, "max": 61, "step": 2, "label": "Block"},
                "C": {"kind": "int", "value": 5, "min": -5, "max": 15, "step": 1, "label": "C"},
                "thresh": {"kind": "int", "value": 127, "min": 0, "max": 255, "step": 1, "label": "Fixed Thresh"},
                "k": {"kind": "float", "value": 0.2, "min": -1.0, "max": 1.0, "step": 0.05, "label": "Sauvola/Niblack K"},
                "blur_ksize": {"kind": "int", "value": 3, "min": 1, "max": 9, "step": 2, "label": "Blur K"},
            },
            description="Convert grayscale ridge image into binary mask.",
            display_key="binary",
            level="core",
        )

    if apply_mask_to_binary is not None:
        ops["mask_binary"] = OperationSpec(
            key="mask_binary",
            name="Apply Mask to Binary",
            func=apply_mask_to_binary,
            input_map=_pick_input_map(
                apply_mask_to_binary,
                {
                    "binary": "binary",
                    "mask": "mask",
                },
            ),
            output_keys=["binary"],
            params_spec={},
            description="Apply ROI mask to binary result.",
            display_key="binary",
            level="core",
        )
    
    # --- Morphology ---
    if apply_morphology is not None:
        ops["morphology"] = OperationSpec(
            key="morphology",
            name="Morphology",
            func=apply_morphology,
            input_map=_pick_input_map(
                apply_morphology,
                {"binary": "binary"},
            ),
            output_keys=["binary"],
            params_spec={
                "open_k": {"kind": "int", "value": 0, "min": 0, "max": 7, "step": 1, "label": "Open K"},
                "close_k": {"kind": "int", "value": 0, "min": 0, "max": 7, "step": 1, "label": "Close K"},
                "dilate_k": {"kind": "int", "value": 0, "min": 0, "max": 3, "step": 1, "label": "Dilate K"},
                "min_area": {"kind": "int", "value": 0, "min": 0, "max": 1000, "step": 10, "label": "Min Area"},
                "fill": {"kind": "bool", "value": False, "label": "Fill Holes"},
            },
            description="Clean small noise and repair binary mask.",
            display_key="binary",
            level="advanced",
        )

    # --- Skeleton ---
    _skeleton_params = {
        "use_skimage":      {"kind": "bool", "value": True,  "label": "Use Scikit-Image"},
        "max_iter":         {"kind": "int",  "value": 100,   "min": 10,  "max": 500, "step": 10,  "label": "Max Iter (CV)"},
        "min_object_size":  {"kind": "int",  "value": 50,    "min": 0,   "max": 200, "step": 10,  "label": "Min Obj Size"},
    }
    _prune_params = {
        "spur_length":          {"kind": "int", "value": 15, "min": 0, "max": 30,  "step": 1, "label": "Spur Length"},
        "min_component_size":   {"kind": "int", "value": 10, "min": 0, "max": 100, "step": 5, "label": "Min Comp Size"},
        "n_passes":             {"kind": "int", "value": 2,  "min": 1, "max": 5,   "step": 1, "label": "Passes"},
    }

    ops["skeleton_vis"] = OperationSpec(
        key="skeleton_vis",
        name="Skeletonization",
        func=step_skeletonize_vis,
        input_map={"image": "binary"},
        output_keys=["skeleton_vis_img", "skeleton"],
        params_spec=_skeleton_params,
        description="Thin binary ridges to 1-pixel skeleton — overlay on image.",
        display_key="skeleton_vis_img",
        level="core",
    )

    ops["skeleton_only_vis"] = OperationSpec(
        key="skeleton_only_vis",
        name="Skeletonization",
        func=step_skeleton_only_vis,
        input_map={"image": "binary"},
        output_keys=["skeleton_vis_img", "skeleton"],
        params_spec=_skeleton_params,
        description="Thin binary ridges to 1-pixel skeleton — white on black.",
        display_key="skeleton_vis_img",
        level="core",
    )

    if prune_skeleton_topology is not None:
        ops["prune_skeleton"] = OperationSpec(
            key="prune_skeleton",
            name="Prune Skeleton",
            func=step_prune_skeleton_vis,
            input_map={"skeleton": "skeleton", "binary_image": "binary"},
            output_keys=["skeleton_vis_img", "skeleton"],
            params_spec=_prune_params,
            description="Remove short spurs / unstable branches — overlay on binary.",
            display_key="skeleton_vis_img",
            level="advanced",
        )
        ops["prune_skeleton_only"] = OperationSpec(
            key="prune_skeleton_only",
            name="Prune Skeleton",
            func=step_prune_skeleton_only_vis,
            input_map={"skeleton": "skeleton"},
            output_keys=["skeleton_vis_img", "skeleton"],
            params_spec=_prune_params,
            description="Remove short spurs / unstable branches — white on black.",
            display_key="skeleton_vis_img",
            level="advanced",
        )

    # --- Minutiae ---
    ops["minutiae_vis"] = OperationSpec(
        key="minutiae_vis",
        name="Minutiae Extraction",
        func=step_minutiae_vis,
        input_map=_pick_input_map(
            step_minutiae_vis,
            {
                "skeleton_img": "skeleton",
                "mask": "mask",
            },
        ),
        output_keys=["minutiae_vis_img", "minutiae_list"], 
        params_spec={
            "border_margin": {"kind": "int", "value": 10, "min": 0, "max": 50, "step": 1, "label": "Border Margin"},
            "cluster_dist": {"kind": "int", "value": 6, "min": 0, "max": 20, "step": 1, "label": "Cluster Dist"},
            "swap_on_inverted_polarity": {"kind": "bool", "value": True, "label": "Swap Polarity"},
            "use_traced_orientation": {"kind": "bool", "value": True, "label": "Traced Orientation"},
            "max_spur_len": {"kind": "int", "value": 15, "min": 0, "max": 30, "step": 1, "label": "Max Spur Len"},
            "suppress_dist": {"kind": "int", "value": 12, "min": 0, "max": 20, "step": 1, "label": "Suppress Dist"},
        },
        description="Detect ridge endings / bifurcations from skeleton and visualize.",
        display_key="minutiae_vis_img",
        level="core",
    )

    return ops


AVAILABLE_OPERATIONS = _available_operation_specs()


# ============================================================
# PRESET DEFINITIONS
# ============================================================

# Minimal, stable chain for scanned fingerprints
SCAN_MINIMAL_ORDER = [
    "roi_segmentation_vis",
    "clahe",
    "binarize",
    "mask_binary",
    "skeleton_vis",
    "minutiae_vis",
]

SCAN_MINIMAL_ENABLED = {
    "roi_segmentation_vis": True,
    "clahe": True,
    "binarize": True,
    "mask_binary": True,
    "skeleton_vis": True,
    "minutiae_vis": True,
}

# Slightly richer scan workflow
SCAN_STUDENT_ORDER = [
    "roi_segmentation_vis",
    "clahe",
    "local_contrast",
    "binarize",
    "mask_binary",
    "morphology",
    "skeleton_vis",
    "prune_skeleton",
    "minutiae_vis",
]

SCAN_STUDENT_ENABLED = {
    "roi_segmentation_vis": True,
    "clahe": True,
    "local_contrast": False,
    "binarize": True,
    "mask_binary": True,
    "morphology": False,
    "skeleton_vis": True,
    "prune_skeleton": False,
    "minutiae_vis": True,
}

# Broader experimentation for photographed fingerprints
PHOTO_HOME_ORDER = [
    "channel_select",
    "mask_black_bg",
    "apply_mask_zero",
    "stretch",
    "gamma",
    "clahe",
    "local_contrast",
    "dog",
    "fft_bandpass",
    "ced",
    "ridge_filter",
    "dynamic_gabor",
    "binarize",
    "mask_binary",
    "morphology",
    "skeleton_vis",
    "prune_skeleton",
    "minutiae_vis",
]

PHOTO_HOME_ENABLED = {
    "channel_select": True,
    "mask_black_bg": True,
    "apply_mask_zero": True,
    "stretch": False,
    "gamma": False,
    "clahe": True,
    "local_contrast": False,
    "dog": False,
    "fft_bandpass": False,
    "ced": False,
    "ridge_filter": False,
    "dynamic_gabor": False,
    "binarize": True,
    "mask_binary": True,
    "morphology": True,
    "skeleton_vis": False,
    "prune_skeleton": False,
    "minutiae_vis": False,
}


# ============================================================
# BUILDERS
# ============================================================

def available_operation_keys(level: Optional[str] = None) -> List[str]:
    if level is None:
        return sorted(AVAILABLE_OPERATIONS.keys())
    return sorted(
        key for key, spec in AVAILABLE_OPERATIONS.items()
        if spec.level == level
    )


def build_pipeline_steps(
    order: List[str],
    enabled_map: Optional[Dict[str, bool]] = None,
    per_step_param_overrides: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
) -> List[Step]:
    enabled_map = enabled_map or {}
    per_step_param_overrides = per_step_param_overrides or {}

    if not isinstance(enabled_map, dict):
        raise TypeError(
            f"enabled_map must be a dict[str, bool], got {type(enabled_map).__name__}"
        )

    steps: List[Step] = []
    for key in order:
        spec = AVAILABLE_OPERATIONS.get(key)
        if spec is None:
            print(f"[build_pipeline_steps] skipping missing operation: {key}")
            continue

        enabled = enabled_map.get(key, True)
        step = _make_step(
            spec,
            enabled=enabled,
            override_params=per_step_param_overrides.get(key),
        )

        if step is None:
            raise RuntimeError(f"_make_step returned None for operation '{key}'")

        steps.append(step)

    return steps


def make_scan_minimal_pipeline(
    image: np.ndarray,
    *,
    preset_name: str = "scan_minimal",
    per_step_param_overrides: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
) -> InteractivePipeline:
    steps = build_pipeline_steps(
        order=SCAN_MINIMAL_ORDER,
        enabled_map=SCAN_MINIMAL_ENABLED,
        per_step_param_overrides=per_step_param_overrides,
    )
    return InteractivePipeline(steps=steps, initial_image=image, preset_name=preset_name)


def make_scan_student_pipeline(
    image: np.ndarray,
    *,
    preset_name: str = "scan_student",
    per_step_param_overrides: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
) -> InteractivePipeline:
    steps = build_pipeline_steps(
        order=SCAN_STUDENT_ORDER,
        enabled_map=SCAN_STUDENT_ENABLED,
        per_step_param_overrides=per_step_param_overrides,
    )
    return InteractivePipeline(steps=steps, initial_image=image, preset_name=preset_name)


def make_photo_home_pipeline(
    image: np.ndarray,
    *,
    preset_name: str = "photo_home",
    per_step_param_overrides: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
) -> InteractivePipeline:
    steps = build_pipeline_steps(
        order=PHOTO_HOME_ORDER,
        enabled_map=PHOTO_HOME_ENABLED,
        per_step_param_overrides=per_step_param_overrides,
    )
    return InteractivePipeline(steps=steps, initial_image=image, preset_name=preset_name)


def make_custom_pipeline(
    image: np.ndarray,
    order: List[str],
    enabled_map: Optional[Dict[str, bool]] = None,
    per_step_param_overrides: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
    *,
    preset_name: str = "custom",
) -> InteractivePipeline:
    steps = build_pipeline_steps(
        order=order,
        enabled_map=enabled_map,
        per_step_param_overrides=per_step_param_overrides,
    )
    return InteractivePipeline(steps=steps, initial_image=image, preset_name=preset_name)
