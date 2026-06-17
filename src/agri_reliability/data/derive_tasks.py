from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from agri_reliability.data.label_mapping import UNIFIED_LABEL_TO_ID, mask_to_unified


def load_mask_as_unified(mask_path: str | Path, config: dict | None = None) -> np.ndarray:
    mask = np.asarray(Image.open(mask_path))
    return mask_to_unified(mask, config)


def derive_image_labels_from_mask(mask: np.ndarray) -> dict:
    mask = np.asarray(mask)
    contains_crop = bool(np.any(mask == UNIFIED_LABEL_TO_ID["crop"]))
    contains_weed = bool(np.any(mask == UNIFIED_LABEL_TO_ID["weed"]))
    return {
        "contains_crop": contains_crop,
        "contains_weed": contains_weed,
        "weed_presence": int(contains_weed),
    }


def boxes_from_mask(mask: np.ndarray, min_area: int = 4) -> list[dict]:
    """Generate boxes from connected crop/weed regions in a unified mask."""
    mask = np.asarray(mask)
    boxes: list[dict] = []
    for label_name in ("crop", "weed"):
        label_id = UNIFIED_LABEL_TO_ID[label_name]
        binary = mask == label_id
        if not np.any(binary):
            continue
        boxes.extend(_connected_component_boxes(binary, label_name, min_area))
    return boxes


def _connected_component_boxes(binary: np.ndarray, label_name: str, min_area: int) -> list[dict]:
    try:
        from scipy import ndimage

        labeled, count = ndimage.label(binary)
        objects = ndimage.find_objects(labeled)
        boxes = []
        for component_id, slices in enumerate(objects, start=1):
            if slices is None:
                continue
            ys, xs = slices
            area = int(np.sum(labeled[slices] == component_id))
            if area < min_area:
                continue
            boxes.append(
                {
                    "label": label_name,
                    "label_id": UNIFIED_LABEL_TO_ID[label_name],
                    "x_min": int(xs.start),
                    "y_min": int(ys.start),
                    "x_max": int(xs.stop - 1),
                    "y_max": int(ys.stop - 1),
                    "area": area,
                    "source": "mask_connected_component",
                }
            )
        return boxes
    except Exception:
        ys, xs = np.where(binary)
        if len(xs) < min_area:
            return []
        return [
            {
                "label": label_name,
                "label_id": UNIFIED_LABEL_TO_ID[label_name],
                "x_min": int(xs.min()),
                "y_min": int(ys.min()),
                "x_max": int(xs.max()),
                "y_max": int(ys.max()),
                "area": int(len(xs)),
                "source": "mask_global_extent",
            }
        ]


def derive_fields_from_mask_path(mask_path: str | Path, config: dict | None = None) -> tuple[dict, list[dict]]:
    mask = load_mask_as_unified(mask_path, config)
    return derive_image_labels_from_mask(mask), boxes_from_mask(mask)
