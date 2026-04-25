from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

try:
    from skimage.filters import frangi, meijering
    SKIMAGE_FILTERS = True
except Exception:
    SKIMAGE_FILTERS = False

from .utils import ensure_odd, normalize_gray


def apply_clahe(img_gray: np.ndarray, clip: float = 2.5, tile: int = 8) -> np.ndarray:
    tile = max(1, int(tile))
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(tile, tile))
    return clahe.apply(img_gray)


def median_blur(img_gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Median blur used before adaptive thresholding in fingerprint pipelines.
    """
    if img_gray.ndim != 2:
        raise ValueError("median_blur expects a 2D grayscale image")

    ksize = ensure_odd(int(ksize), 3)
    if ksize < 3:
        raise ValueError("ksize must be >= 3")

    return cv2.medianBlur(img_gray, ksize)

def clahe_median_pipeline(
    gray: np.ndarray,
    clahe_clip: float = 2.0,
    clahe_tile: int = 8,
    median_ksize: int = 3,
) -> np.ndarray:
    """
    Preprocessing used in detector script:
        CLAHE → median blur
    """
    enhanced = apply_clahe(gray, clip=clahe_clip, tile=clahe_tile)
    enhanced = median_blur(enhanced, ksize=median_ksize)
    return enhanced


def contrast_stretch_saturated(img_gray: np.ndarray, saturated: float = 0.35) -> np.ndarray:
    img = img_gray.astype(np.float32)
    lo = np.percentile(img, saturated)
    hi = np.percentile(img, 100.0 - saturated)
    if hi <= lo:
        return img_gray.copy()
    return np.clip((img - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)


def gamma_correction(img_gray: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    gamma = max(gamma, 1e-6)
    x = img_gray.astype(np.float32) / 255.0
    out = np.power(x, gamma)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def local_contrast_enhancement(img_gray: np.ndarray, radius: int = 31, amount: float = 1.0) -> np.ndarray:
    radius = ensure_odd(radius, 3)
    local = cv2.GaussianBlur(img_gray, (radius, radius), 0)
    out = img_gray.astype(np.float32) + float(amount) * (img_gray.astype(np.float32) - local.astype(np.float32))
    return normalize_gray(out)


def difference_of_gaussians(img: np.ndarray, sigma1: float = 1.0, sigma2: float = 8.0, gain: float = 1.5) -> np.ndarray:
    imgf = img.astype(np.float32) / 255.0
    b1 = cv2.GaussianBlur(imgf, (0, 0), sigmaX=float(sigma1), sigmaY=float(sigma1))
    b2 = cv2.GaussianBlur(imgf, (0, 0), sigmaX=float(sigma2), sigmaY=float(sigma2))
    dog = (b1 - b2) * float(gain)
    dog = dog - dog.min()
    if dog.max() > 0:
        dog = dog / dog.max()
    return np.clip(dog * 255.0, 0, 255).astype(np.uint8)


def fft_gaussian_bandpass(img: np.ndarray, low_sigma: float = 3.0, high_sigma: float = 28.0) -> np.ndarray:
    imgf = img.astype(np.float32)
    h, w = imgf.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r2 = (y - cy) ** 2 + (x - cx) ** 2
    lowpass_small = np.exp(-r2 / (2.0 * float(low_sigma) ** 2))
    lowpass_large = np.exp(-r2 / (2.0 * float(high_sigma) ** 2))
    bandpass = ((1.0 - lowpass_small) * lowpass_large).astype(np.float32)
    f = np.fft.fftshift(np.fft.fft2(imgf))
    g = f * bandpass
    out = np.real(np.fft.ifft2(np.fft.ifftshift(g)))
    return normalize_gray(out)


def gabor_rebuild(img_gray: np.ndarray, sigma: float, freq: float, kernel_size: int = 21,
                  gamma: float = 0.5, n_angles: int = 16) -> np.ndarray:
    kernel_size = ensure_odd(kernel_size, 3)
    combined = np.zeros_like(img_gray, dtype=np.float32)
    lambd = max(1.0 / max(freq, 1e-6), 1.0)
    for theta in np.linspace(0, np.pi, n_angles, endpoint=False):
        kernel = cv2.getGaborKernel((kernel_size, kernel_size), float(sigma), float(theta), float(lambd), float(gamma), 0, ktype=cv2.CV_32F)
        filtered = cv2.filter2D(img_gray, cv2.CV_32F, kernel)
        combined = np.maximum(combined, filtered)
    return normalize_gray(combined)


def estimate_local_orientation(gray: np.ndarray, block: int = 16, smooth_ksize: int = 5) -> np.ndarray:
    gray_f = gray.astype(np.float32)
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    gxx, gyy, gxy = gx * gx, gy * gy, gx * gy
    win = (max(1, int(block)), max(1, int(block)))
    gxx = cv2.boxFilter(gxx, -1, win, normalize=True)
    gyy = cv2.boxFilter(gyy, -1, win, normalize=True)
    gxy = cv2.boxFilter(gxy, -1, win, normalize=True)
    theta = 0.5 * np.arctan2(2.0 * gxy, (gxx - gyy) + 1e-8)
    smooth_ksize = ensure_odd(smooth_ksize, 1)
    if smooth_ksize > 1:
        c = cv2.GaussianBlur(np.cos(2.0 * theta), (smooth_ksize, smooth_ksize), 0)
        s = cv2.GaussianBlur(np.sin(2.0 * theta), (smooth_ksize, smooth_ksize), 0)
        theta = 0.5 * np.arctan2(s, c)
    return theta.astype(np.float32)


def estimate_orientation_coherence(gray: np.ndarray, block: int = 16) -> np.ndarray:
    gray_f = gray.astype(np.float32)
    gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    gxx, gyy, gxy = gx * gx, gy * gy, gx * gy
    win = (max(1, int(block)), max(1, int(block)))
    gxx = cv2.boxFilter(gxx, -1, win, normalize=True)
    gyy = cv2.boxFilter(gyy, -1, win, normalize=True)
    gxy = cv2.boxFilter(gxy, -1, win, normalize=True)
    num = np.sqrt((gxx - gyy) ** 2 + 4.0 * gxy ** 2)
    den = gxx + gyy + 1e-8
    return np.clip(num / den, 0.0, 1.0).astype(np.float32)


def estimate_block_frequency(block_img: np.ndarray, block_theta: float, min_wave: int = 4, max_wave: int = 20) -> Optional[float]:
    h, w = block_img.shape
    cy, cx = h // 2, w // 2
    perp = float(block_theta) + np.pi / 2.0
    dy, dx = np.sin(perp), np.cos(perp)
    half_len = min(h, w) // 2 - 1
    vals = []
    for t in range(-half_len, half_len + 1):
        y = int(round(cy + t * dy))
        x = int(round(cx + t * dx))
        if 0 <= y < h and 0 <= x < w:
            vals.append(float(block_img[y, x]))
    vals = np.asarray(vals, dtype=np.float32)
    if len(vals) < max_wave * 2:
        return None
    vals = vals - vals.mean()
    std = float(vals.std())
    if std < 1e-6:
        return None
    vals = vals / std
    vals_sm = cv2.GaussianBlur(vals.reshape(1, -1), (1, 5), 0).ravel()
    peaks = [i for i in range(1, len(vals_sm) - 1) if vals_sm[i] > vals_sm[i - 1] and vals_sm[i] >= vals_sm[i + 1] and vals_sm[i] > 0]
    if len(peaks) < 2:
        return None
    dists = np.diff(peaks).astype(np.float32)
    dists = dists[(dists >= min_wave) & (dists <= max_wave)]
    if len(dists) == 0:
        return None
    wavelength = float(np.median(dists))
    return None if wavelength <= 0 else 1.0 / wavelength


def estimate_frequency_map(gray: np.ndarray, theta: np.ndarray, block: int = 32, min_wave: int = 4,
                           max_wave: int = 20, default_freq: float = 0.08) -> np.ndarray:
    h, w = gray.shape
    block = max(4, int(block))
    freq_map = np.full((h, w), float(default_freq), dtype=np.float32)
    for y0 in range(0, h, block):
        for x0 in range(0, w, block):
            y1, x1 = min(y0 + block, h), min(x0 + block, w)
            patch = gray[y0:y1, x0:x1]
            if patch.size == 0:
                continue
            theta_mean = float(np.median(theta[y0:y1, x0:x1]))
            freq = estimate_block_frequency(patch, theta_mean, min_wave=min_wave, max_wave=max_wave)
            freq_map[y0:y1, x0:x1] = float(default_freq if freq is None else freq)
    return freq_map


def adaptive_gabor_enhance(gray: np.ndarray, theta_map: np.ndarray, freq_map: np.ndarray, block: int = 32,
                           sigma_base: float = 4.0, gamma: float = 0.5, kernel_size: int = 21) -> np.ndarray:
    gray_f = gray.astype(np.float32)
    h, w = gray.shape
    block = max(4, int(block))
    kernel_size = ensure_odd(kernel_size, 3)
    acc = np.zeros((h, w), dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)
    for y0 in range(0, h, block):
        for x0 in range(0, w, block):
            y1, x1 = min(y0 + block, h), min(x0 + block, w)
            patch = gray_f[y0:y1, x0:x1]
            if patch.size == 0:
                continue
            local_theta = float(np.median(theta_map[y0:y1, x0:x1]))
            local_freq = max(float(np.median(freq_map[y0:y1, x0:x1])), 1e-3)
            lambd = max(1.0 / local_freq, 1.0)
            sigma = max(1.0, float(sigma_base) * (lambd / 10.0))
            kernel = cv2.getGaborKernel((kernel_size, kernel_size), sigma, local_theta, lambd, float(gamma), 0, ktype=cv2.CV_32F)
            filtered = cv2.filter2D(patch, cv2.CV_32F, kernel)
            acc[y0:y1, x0:x1] += filtered
            weight[y0:y1, x0:x1] += 1.0
    out = acc / np.maximum(weight, 1e-6)
    return normalize_gray(out)


def dynamic_gabor_pipeline(gray: np.ndarray, orient_block: int = 16, orient_smooth: int = 5, freq_block: int = 32,
                           min_wave: int = 4, max_wave: int = 20, default_freq: float = 0.08,
                           sigma_base: float = 4.0, gamma: float = 0.5, kernel_size: int = 21) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta_map = estimate_local_orientation(gray, block=orient_block, smooth_ksize=orient_smooth)
    freq_map = estimate_frequency_map(gray, theta_map, block=freq_block, min_wave=min_wave, max_wave=max_wave, default_freq=default_freq)
    enhanced = adaptive_gabor_enhance(gray, theta_map, freq_map, block=freq_block, sigma_base=sigma_base, gamma=gamma, kernel_size=kernel_size)
    return enhanced, theta_map, freq_map


def coherence_enhancing_diffusion(img: np.ndarray, lambda_: float = 0.1, sigma: float = 1.0, rho: float = 3.0,
                                  step_size: float = 0.2, m: float = 1.0, n_steps: int = 5) -> np.ndarray:
    """
    Lightweight approximation of coherence-enhancing diffusion.

    This is not a faithful implementation of Weickert CED. It uses repeated
    anisotropic-like Gaussian smoothing as a practical substitute for
    interactive experimentation.
    """

    out = img.astype(np.float32)
    for _ in range(max(1, int(n_steps))):
        blur1 = cv2.GaussianBlur(out, (0, 0), sigmaX=max(0.1, sigma), sigmaY=max(0.1, sigma))
        blur2 = cv2.GaussianBlur(out, (0, 0), sigmaX=max(0.1, rho), sigmaY=max(0.1, sigma))
        out = (1.0 - step_size) * out + step_size * ((1.0 - lambda_) * blur1 + lambda_ * blur2)
    return normalize_gray(out)


def apply_ridge_filter(img_gray: np.ndarray, method: str = "frangi", sigmas: Tuple[float, ...] = (1.0, 2.0, 3.0),
                       black_ridges: bool = True) -> np.ndarray:
    if method == "none":
        return img_gray.copy()
    if not SKIMAGE_FILTERS:
        return img_gray.copy()
    imgf = img_gray.astype(np.float32) / 255.0
    if method == "frangi":
        out = frangi(imgf, sigmas=list(sigmas), black_ridges=bool(black_ridges))
    elif method == "meijering":
        out = meijering(imgf, sigmas=list(sigmas), black_ridges=bool(black_ridges))
    else:
        raise ValueError(f"Unknown ridge_filter method: {method}")
    return normalize_gray(out)


def subtract_background_image(
    img: np.ndarray,
    background: np.ndarray,
) -> np.ndarray:
    """
    Subtract a precomputed background/smoothed image from the original.
    """
    if img.ndim != 2 or background.ndim != 2:
        raise ValueError("subtract_background_image expects 2D grayscale images")
    if img.shape != background.shape:
        raise ValueError("img and background must have the same shape")

    return cv2.subtract(img, background)

def suppress_ridges_with_median(
    img_gray: np.ndarray,
    ksize: int = 31,
) -> np.ndarray:
    """
    Heavy median smoothing to suppress ridge/valley structure and keep only
    coarse background / illumination.
    """
    if img_gray.ndim != 2:
        raise ValueError("suppress_ridges_with_median expects a 2D grayscale image")

    ksize = ensure_odd(int(ksize), 3)
    if ksize < 3:
        raise ValueError("ksize must be >= 3")

    return cv2.medianBlur(img_gray, ksize)

def subtract_heavy_median(
    img_gray: np.ndarray,
    ksize: int = 31,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Heavy median ridge suppression followed by subtraction from the original.
    Returns:
        subtracted, heavy_median
    """
    heavy = suppress_ridges_with_median(img_gray, ksize=ksize)
    sub = subtract_background_image(img_gray, heavy)
    return sub, heavy

try:
    from skimage.restoration import rolling_ball
    HAVE_ROLLING_BALL = True
except Exception:
    HAVE_ROLLING_BALL = False
def subtract_rolling_background(
    img_gray: np.ndarray,
    radius: int = 25,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Approximate ImageJ 'Subtract Background...' for grayscale fingerprint images.

    Returns
    -------
    subtracted : np.ndarray
        Original image minus estimated smooth background.
    background : np.ndarray
        Estimated background image.
    """
    if img_gray.ndim != 2:
        raise ValueError("subtract_rolling_background expects a 2D grayscale image")

    radius = max(1, int(radius))

    if HAVE_ROLLING_BALL:
        background = rolling_ball(img_gray, radius=radius).astype(np.uint8)
    else:
        ksize = ensure_odd(2 * radius + 1, 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        background = cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, kernel)

    subtracted = cv2.subtract(img_gray, background)
    return subtracted, background


def subtract_median_ridges(gray: np.ndarray, finger_scale: float, base_radius: int = 15):
    """
    Equivalent to ImageJ 'MedianWithoutRidgesSubtracted'
    Removes the ridge/valley structure to isolate global illumination.
    """
    radius = int(base_radius * finger_scale)
    if radius % 2 == 0: radius += 1

    # Extract background (ridges removed)
    background = cv2.medianBlur(gray, radius)

    # Subtract background from original
    subtracted = cv2.subtract(gray, background)

    # Enhance contrast like ImageJ 'saturated=0.35'
    return contrast_stretch_saturated(subtracted, saturated=0.35)


def homomorphic_filter(img: np.ndarray, cutoff: int = 30) -> np.ndarray:
    """Removes uneven illumination (flash hotspots)."""
    img_log = np.log1p(img.astype(np.float32))
    dft = np.fft.fftshift(np.fft.fft2(img_log))

    rows, cols = img.shape
    mask = np.ones((rows, cols))
    cy, cx = rows // 2, cols // 2
    for i in range(rows):
        for j in range(cols):
            dist = np.sqrt((i - cy) ** 2 + (j - cx) ** 2)
            mask[i, j] = 1 - np.exp(-(dist ** 2) / (2 * (cutoff ** 2)))

    f_filt = dft * (0.5 + mask)  # Boost highs, suppress lows
    img_back = np.expm1(np.real(np.fft.ifft2(np.fft.ifftshift(f_filt))))
    return cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def local_normalize(img: np.ndarray, block_size: int = 16):
    """Hong-Wan-Jain normalization for uniform contrast."""
    img = img.astype(np.float32)
    mean = cv2.blur(img, (block_size, block_size))
    sq_mean = cv2.blur(img ** 2, (block_size, block_size))
    std = np.sqrt(np.maximum(sq_mean - mean ** 2, 10))
    norm = (img - mean) / std
    return cv2.normalize(norm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)