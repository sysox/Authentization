import cv2
import numpy as np
from typing import Tuple


def rectify_perspective(img: np.ndarray, src_pts: np.ndarray) -> np.ndarray:
    """
    Warps the image to a top-down view based on 4 detected landmarks.
    """
    # Define destination points as a standard rectangle
    width = int(max(np.linalg.norm(src_pts[0] - src_pts[1]), np.linalg.norm(src_pts[2] - src_pts[3])))
    height = int(max(np.linalg.norm(src_pts[0] - src_pts[3]), np.linalg.norm(src_pts[1] - src_pts[2])))

    dst_pts = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
    matrix = cv2.getPerspectiveTransform(src_pts.astype("float32"), dst_pts)
    return cv2.warpPerspective(img, matrix, (width, height))


def cylindrical_unwrap(img: np.ndarray, focal_length: float = 800) -> np.ndarray:
    """
    Corrects the 'squashing' effect at the edges of a photographed finger.
    """
    h, w = img.shape[:2]
    K = np.array([[focal_length, 0, w / 2], [0, focal_length, h / 2], [0, 0, 1]])
    K_inv = np.linalg.inv(K)

    y_i, x_i = np.indices((h, w))
    coords = np.stack([x_i.ravel(), y_i.ravel(), np.ones(h * w)], axis=-1)
    norm_coords = K_inv.dot(coords.T).T

    # Map to cylinder coordinates
    x_cyl = np.sin(norm_coords[:, 0])
    y_cyl = norm_coords[:, 1]
    z_cyl = np.cos(norm_coords[:, 0])

    # Project back to 2D plane
    A = np.stack([x_cyl, y_cyl, z_cyl], axis=-1)
    B = K.dot(A.T).T
    B = B[:, :-1] / B[:, [-1]]

    map_x = B[:, 0].reshape(h, w).astype(np.float32)
    map_y = B[:, 1].reshape(h, w).astype(np.float32)
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)