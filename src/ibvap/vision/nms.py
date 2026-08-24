"""Non-maximum suppression in pure NumPy.

Implemented here rather than pulled from torchvision so that the runtime
dependency footprint stays at NumPy + ONNX Runtime. At the box counts a CCTV
frame produces (tens, occasionally low hundreds) the vectorised NumPy loop is
comfortably faster than the cost of importing a deep-learning framework.
"""

from __future__ import annotations

import numpy as np


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45) -> np.ndarray:
    """Greedy NMS over ``(N, 4)`` xyxy boxes. Returns kept indices, best first."""
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.int64)

    boxes = boxes.astype(np.float64, copy=False)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        best = order[0]
        keep.append(int(best))
        if order.size == 1:
            break
        rest = order[1:]

        ix1 = np.maximum(x1[best], x1[rest])
        iy1 = np.maximum(y1[best], y1[rest])
        ix2 = np.minimum(x2[best], x2[rest])
        iy2 = np.minimum(y2[best], y2[rest])
        inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
        union = areas[best] + areas[rest] - inter
        iou = np.where(union > 1e-9, inter / np.maximum(union, 1e-9), 0.0)

        order = rest[iou <= iou_threshold]

    return np.asarray(keep, dtype=np.int64)


def batched_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float = 0.45,
) -> np.ndarray:
    """Class-aware NMS: boxes of different classes never suppress each other.

    This matters in practice - a motorcyclist's ``person`` box overlaps their
    ``motorcycle`` box almost completely, and class-agnostic NMS would delete
    one of them. At a border check post that is the difference between
    "one vehicle" and "one vehicle carrying one person".
    """
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.int64)

    # Shift each class into its own coordinate band so IoU across classes is 0.
    max_extent = float(boxes.max()) if boxes.size else 1.0
    offsets = class_ids.astype(np.float64) * (max_extent + 1.0)
    shifted = boxes + offsets[:, None]
    return nms(shifted, scores, iou_threshold)


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert ``(N, 4)`` centre-form boxes to corner form."""
    out = np.empty_like(boxes, dtype=np.float64)
    half_w, half_h = boxes[:, 2] / 2.0, boxes[:, 3] / 2.0
    out[:, 0] = boxes[:, 0] - half_w
    out[:, 1] = boxes[:, 1] - half_h
    out[:, 2] = boxes[:, 0] + half_w
    out[:, 3] = boxes[:, 1] + half_h
    return out


def box_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between ``(N, 4)`` and ``(M, 4)`` xyxy boxes -> ``(N, M)``."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)

    a = a.astype(np.float64, copy=False)
    b = b.astype(np.float64, copy=False)
    area_a = np.maximum(0.0, a[:, 2] - a[:, 0]) * np.maximum(0.0, a[:, 3] - a[:, 1])
    area_b = np.maximum(0.0, b[:, 2] - b[:, 0]) * np.maximum(0.0, b[:, 3] - b[:, 1])

    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)

    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 1e-9, inter / np.maximum(union, 1e-9), 0.0)
