from __future__ import annotations

import numpy as np


def segmentation_confusion_matrix(y_true, y_pred, num_classes: int = 3) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_pred = np.asarray(y_pred, dtype=int).ravel()
    mask = (y_true >= 0) & (y_true < num_classes)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    y_pred = np.clip(y_pred, 0, num_classes - 1)
    return np.bincount(
        num_classes * y_true + y_pred,
        minlength=num_classes**2,
    ).reshape(num_classes, num_classes)


def class_iou_from_confusion(confusion: np.ndarray) -> dict[str, float]:
    confusion = np.asarray(confusion, dtype=float)
    intersection = np.diag(confusion)
    ground_truth = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    union = ground_truth + predicted - intersection
    iou = np.divide(
        intersection,
        union,
        out=np.full_like(intersection, np.nan, dtype=float),
        where=union > 0,
    )
    names = ["background", "crop", "weed"]
    return {f"iou_{names[idx]}": float(value) for idx, value in enumerate(iou)}


def mean_iou(y_true, y_pred, num_classes: int = 3, ignore_background: bool = False) -> float:
    confusion = segmentation_confusion_matrix(y_true, y_pred, num_classes)
    values = np.asarray(list(class_iou_from_confusion(confusion).values()), dtype=float)
    if ignore_background and len(values) > 1:
        values = values[1:]
    return float(np.nanmean(values)) if np.any(~np.isnan(values)) else float("nan")


def segmentation_summary(y_true, y_pred, num_classes: int = 3) -> dict[str, float]:
    confusion = segmentation_confusion_matrix(y_true, y_pred, num_classes)
    metrics = class_iou_from_confusion(confusion)
    metrics["miou"] = mean_iou(y_true, y_pred, num_classes)
    metrics["foreground_miou"] = mean_iou(y_true, y_pred, num_classes, ignore_background=True)
    return metrics
