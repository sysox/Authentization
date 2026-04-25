from __future__ import annotations

import numpy as np

def estimate_image_quality(gray: np.ndarray, mask: np.ndarray | None = None, block_size: int = 32) -> float:
    """
    Estimates the overall quality of a fingerprint image.
    
    A higher score indicates better quality.
    The score is based on the strength of the frequency content in local blocks.
    A clear ridge pattern will have a strong peak in the frequency domain.
    """
    h, w = gray.shape
    scores = []
    
    step = block_size
    for y in range(0, h - block_size + 1, step):
        for x in range(0, w - block_size + 1, step):
            if mask is not None:
                patch_mask = mask[y:y+block_size, x:x+block_size]
                if np.count_nonzero(patch_mask) < (block_size * block_size * 0.5):
                    continue
            
            patch = gray[y:y+block_size, x:x+block_size]
            
            # Simple quality metric: Ratio of energy in dominant frequency to total energy
            # (excluding DC)
            patch_centered = patch - np.mean(patch)
            f = np.fft.fft2(patch_centered)
            fshift = np.fft.fftshift(f)
            psd = np.abs(fshift)**2
            
            cy, cx = block_size // 2, block_size // 2
            psd[cy, cx] = 0 # Remove DC
            
            total_energy = np.sum(psd)
            if total_energy < 1e-6:
                scores.append(0.0)
                continue
                
            peak_energy = np.max(psd)
            scores.append(peak_energy / total_energy)
            
    if not scores:
        return 0.0
        
    return float(np.mean(scores))
