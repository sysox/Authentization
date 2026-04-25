from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Literal


@dataclass
class PipelineParams:

    # ------------------------------------------------------------
    # INPUT
    # ------------------------------------------------------------
    resize_to: Optional[int] = None
    invert_input: bool = False

    # ------------------------------------------------------------
    # MASK / ROI
    # ------------------------------------------------------------
    apply_mask: bool = True
    auto_mask: bool = True
    use_variance_roi: bool = False

    # variance ROI parameters
    roi_block: int = 16
    roi_close_k: int = 25
    roi_erode_k: int = 7
    roi_close_iter: int = 4
    roi_dilate_iter: int = 1

    # black background masking
    mask_from_black_background: bool = True
    black_bg_thresh: int = 25

    # foreground mask
    mask_blur_k: int = 9
    mask_block: int = 31
    mask_C: int = 5

    mask_open_k: int = 5
    mask_close_k: int = 9
    mask_min_area: int = 2000

    # ------------------------------------------------------------
    # ENHANCEMENT
    # ------------------------------------------------------------
    clahe_clip: float = 2.0
    clahe_tile: int = 8

    gamma: float = 1.0

    local_contrast_radius: int = 0
    local_contrast_amount: float = 1.0

    median_blur_k: int = 0
    blur_k: int = 3

    # ------------------------------------------------------------
    # DoG / FFT
    # ------------------------------------------------------------
    # "dog", "fft", "gabor", "ced", "none"
    enhancement_method: Literal["none", "dog", "fft", "gabor", "ced"] = "none"

    dog_sigma1: float = 1.0
    dog_sigma2: float = 2.0
    dog_gain: float = 1.0

    fft_low_sigma: float = 3.0
    fft_high_sigma: float = 25.0

    # ------------------------------------------------------------
    # GABOR
    # ------------------------------------------------------------
    use_dynamic_gabor: bool = False

    gabor_sigma: float = 3.0
    gabor_gamma: float = 0.5
    gabor_kernel: int = 15
    gabor_angles: int = 8
    gabor_freq: float = 0.1

    orient_block: int = 16
    orient_smooth: int = 3

    freq_block: int = 32
    freq_min_wave: int = 5
    freq_max_wave: int = 15
    freq_default: int = 10

    # ------------------------------------------------------------
    # CED (Coherence Enhancing Diffusion)
    # ------------------------------------------------------------
    ced_lambda: float = 0.2
    ced_sigma: float = 1.0
    ced_rho: float = 3.0
    ced_step_size: float = 0.15
    ced_m: int = 1
    ced_steps: int = 5

    # ------------------------------------------------------------
    # RIDGE FILTER
    # ------------------------------------------------------------
    ridge_filter: Literal["none", "frangi", "sato", "meijering", "hessian"] = "none"
    ridge_sigmas: Tuple[float, ...] = (1.0, 2.0, 3.0)
    black_ridges: bool = False

    # ------------------------------------------------------------
    # BINARIZATION
    # ------------------------------------------------------------
    bin_method: Literal["adaptive", "otsu", "sauvola", "niblack", "fixed"] = "adaptive"
    bin_block: int = 15
    bin_C: int = 3
    bin_thresh: int = 127
    bin_k: float = 0.2

    # detector-specific extras
    bin_blur_ksize: int = 3
    bin_invert_before_check: bool = True
    want_ridges_white: bool = True

    # ------------------------------------------------------------
    # MORPHOLOGY
    # ------------------------------------------------------------
    close_k: int = 3
    open_k: int = 2
    morph_min_area: int = 0
    fill_holes: bool = False
    dilate_k: int = 0

    # ------------------------------------------------------------
    # SKELETON
    # ------------------------------------------------------------
    use_skimage_skeletonize: bool = True
    skeleton_max_iter: int = 100
    skeleton_min_object_size: int = 50

    prune_spurs: bool = True
    spur_length: int = 15
    prune_passes: int = 2

    min_skeleton_component: int = 10

    # ------------------------------------------------------------
    # MINUTIAE
    # ------------------------------------------------------------
    extract_minutiae: bool = True
    finalize_minutiae: bool = True

    minutiae_border_margin: int = 10
    minutiae_cluster_dist: int = 6

    minutiae_max_spur_len: int = 15
    minutiae_suppress_dist: int = 12

    def __post_init__(self):
        """
        Validates and corrects parameters after initialization.
        Ensures kernel sizes are odd for OpenCV compatibility.
        """
        # Ensure kernel sizes are odd numbers
        for attr in ['roi_block', 'mask_blur_k', 'mask_block', 'mask_open_k',
                     'mask_close_k', 'blur_k', 'gabor_kernel', 'orient_block',
                     'bin_block', 'close_k', 'open_k']:
            val = getattr(self, attr)
            if val is not None and val > 0 and val % 2 == 0:
                setattr(self, attr, val + 1)