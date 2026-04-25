from __future__ import annotations

import cv2
import numpy as np
from typing import Optional


def _calculate_freq_from_patch(patch: np.ndarray) -> float:
    h, w = patch.shape
    if h < 8 or w < 8:
        return 0.0
    # Remove DC component
    patch_centered = patch - np.mean(patch)
    f = np.fft.fft2(patch_centered)
    fshift = np.fft.fftshift(f)
    psd = np.abs(fshift)**2
    
    cy, cx = h // 2, w // 2
    psd[cy, cx] = 0  # Zero out DC
    
    # Find peak
    idx = np.unravel_index(np.argmax(psd), psd.shape)
    dist = np.sqrt((idx[0] - cy)**2 + (idx[1] - cx)**2)
    
    # frequency = dist / size
    # period (wavelength) = size / dist
    return float(h) / dist if dist > 0 else 0.0


def estimate_ridge_frequency(img: np.ndarray, block_size: int = 32) -> float:
    """
    Estimates the average pixels between ridges using FFT on a central block.
    Returns the wavelength (period) in pixels.
    """
    rows, cols = img.shape
    cy, cx = rows // 2, cols // 2
    half = block_size // 2
    
    # Extract central block. Handle boundary conditions if image is small.
    y0 = max(0, cy - half)
    x0 = max(0, cx - half)
    y1 = min(rows, y0 + block_size)
    x1 = min(cols, x0 + block_size)
    
    blk = img[y0:y1, x0:x1]
    
    if blk.shape[0] != block_size or blk.shape[1] != block_size:
        # If image is smaller than block_size, use what we have
        pass
        
    return _calculate_freq_from_patch(blk)


def normalize_to_dpi(img: np.ndarray, current_freq: float, target_dpi: int = 500) -> np.ndarray:
    """
    Rescales the image so morphology operations (like spur removal) remain consistent.
    Standard ridge distance at 500dpi is ~10 pixels.
    """
    if current_freq <= 0:
        return img
        
    target_freq = 10.0
    scale = target_freq / current_freq
    if abs(scale - 1.0) < 0.01:
        return img
        
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def auto_detect_polarity(img_gray: np.ndarray, mask: Optional[np.ndarray] = None) -> bool:
    """
    Analyzes ridge intensity vs background intensity.
    Returns True if ridges are darker than valleys (Standard Scan).
    Returns False if ridges are lighter than valleys (Optical/Latent).
    """
    if mask is not None:
        vals = img_gray[mask > 0]
    else:
        vals = img_gray.flatten()
    
    if vals.size == 0:
        return True
        
    mean = np.mean(vals)
    std = np.std(vals)
    
    if std < 1e-6:
        # Flat image, assume standard
        return True
        
    # Calculate skewness: E[(x-mu)^3] / sigma^3
    skew = np.mean(((vals - mean) / std) ** 3)
    
    # Negative skew -> Mass is on the right (bright background), Tail on the left (dark ridges)
    # This corresponds to Standard Scan (True).
    return bool(skew < 0)


def auto_detect_scale(img_gray: np.ndarray, mask: Optional[np.ndarray] = None, block_size: int = 32) -> float:
    """
    Returns the average ridge period (pixels per ridge) across the image.
    """
    h, w = img_gray.shape
    vals = []
    
    # Sample blocks across the image
    step = block_size
    for y in range(0, h - block_size + 1, step):
        for x in range(0, w - block_size + 1, step):
            # If mask is provided, ensure block is mostly valid
            if mask is not None:
                patch_mask = mask[y:y+block_size, x:x+block_size]
                if np.count_nonzero(patch_mask) < (block_size * block_size * 0.5):
                    continue
            
            patch = img_gray[y:y+block_size, x:x+block_size]
            freq = _calculate_freq_from_patch(patch)
            if freq > 2.0: # Filter out very high frequencies (noise) or DC (0)
                vals.append(freq)
    
    if not vals:
        # Fallback to single center estimation
        return estimate_ridge_frequency(img_gray, block_size)
        
    return float(np.median(vals))
