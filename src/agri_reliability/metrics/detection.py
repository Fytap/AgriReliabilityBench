from __future__ import annotations

import numpy as np


def box_iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1 + 1.0), max(0.0, iy2 - iy1 + 1.0)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1 + 1.0) * max(0.0, ay2 - ay1 + 1.0)
    area_b = max(0.0, bx2 - bx1 + 1.0) * max(0.0, by2 - by1 + 1.0)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def precision_recall_at_iou(y_true_boxes, y_pred_boxes, iou_threshold: float = 0.5) -> dict[str, float]:
    """Small detection smoke metric for one class-agnostic box set."""
    matched_true = set()
    tp = 0
    pred_sorted = sorted(y_pred_boxes, key=lambda item: float(item.get("score", 1.0)), reverse=True)
    true_boxes = [_to_xyxy(box) for box in y_true_boxes]

    for pred in pred_sorted:
        pred_box = _to_xyxy(pred)
        best_iou, best_idx = 0.0, None
        for idx, true_box in enumerate(true_boxes):
            if idx in matched_true:
                continue
            iou = box_iou(pred_box, true_box)
            if iou > best_iou:
                best_iou, best_idx = iou, idx
        if best_idx is not None and best_iou >= iou_threshold:
            tp += 1
            matched_true.add(best_idx)

    fp = max(0, len(y_pred_boxes) - tp)
    fn = max(0, len(y_true_boxes) - tp)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": float(precision), "recall": float(recall), "ap50_proxy": float(precision * recall)}


def _to_xyxy(box) -> tuple[float, float, float, float]:
    if isinstance(box, dict):
        return (
            float(box["x_min"]),
            float(box["y_min"]),
            float(box["x_max"]),
            float(box["y_max"]),
        )
    arr = np.asarray(box, dtype=float).ravel()
    return float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])
