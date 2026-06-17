from __future__ import annotations

from collections.abc import Mapping

import numpy as np


UNIFIED_LABELS = ["background", "crop", "weed"]
UNIFIED_LABEL_TO_ID = {name: idx for idx, name in enumerate(UNIFIED_LABELS)}


def unified_label_space() -> list[str]:
    return list(UNIFIED_LABELS)


def normalize_label_name(name: str | None) -> str:
    if name is None:
        return ""
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


def label_name_to_unified(label_name: str | None, config: Mapping | None = None) -> str | None:
    """Map a dataset-specific class name to crop/weed/background.

    The mapping is intentionally configurable because CropAndWeed and
    PhenoBench may expose either coarse masks or fine-grained plant labels.
    """
    if not label_name:
        return None

    normalized = normalize_label_name(label_name)
    mapping = {}
    keyword_rules = {}
    if config:
        mapping = {
            normalize_label_name(key): normalize_label_name(value)
            for key, value in (config.get("label_name_mapping") or {}).items()
        }
        keyword_rules = config.get("label_keywords") or {}

    explicit = mapping.get(normalized)
    if explicit in UNIFIED_LABEL_TO_ID:
        return explicit

    for unified_name in UNIFIED_LABELS:
        if normalized == unified_name:
            return unified_name

    for unified_name, keywords in keyword_rules.items():
        if normalize_label_name(unified_name) not in UNIFIED_LABEL_TO_ID:
            continue
        for keyword in keywords or []:
            if normalize_label_name(keyword) in normalized:
                return normalize_label_name(unified_name)

    if any(token in normalized for token in ("weed", "weedling", "unwanted")):
        return "weed"
    if any(
        token in normalized
        for token in (
            "crop",
            "maize",
            "corn",
            "sugarbeet",
            "beet",
            "soy",
            "soybean",
            "sunflower",
            "canola",
            "rapeseed",
            "wheat",
            "barley",
        )
    ):
        return "crop"
    if any(token in normalized for token in ("background", "soil", "void", "ignore")):
        return "background"
    return None


def label_id_to_unified(label_id: int | str, config: Mapping | None = None) -> str | None:
    if isinstance(label_id, str) and not label_id.isdigit():
        return label_name_to_unified(label_id, config)

    value = int(label_id)
    mask_values = (config or {}).get("mask_label_values") or {}
    for unified_name, values in mask_values.items():
        if normalize_label_name(unified_name) not in UNIFIED_LABEL_TO_ID:
            continue
        if value in [int(v) for v in values or []]:
            return normalize_label_name(unified_name)

    if value in (0, 1, 2):
        return UNIFIED_LABELS[value]
    return None


def mask_to_unified(mask: np.ndarray, config: Mapping | None = None) -> np.ndarray:
    """Convert a semantic mask to IDs: 0 background, 1 crop, 2 weed.

    Unknown values are treated as background by default. For RGB masks, each
    channel triplet can be configured through ``mask_color_values``.
    """
    mask = np.asarray(mask)
    out = np.zeros(mask.shape[:2], dtype=np.uint8)

    color_values = (config or {}).get("mask_color_values") or {}
    if mask.ndim == 3 and color_values:
        rgb = mask[..., :3].astype(np.uint8)
        for unified_name, colors in color_values.items():
            unified_name = normalize_label_name(unified_name)
            if unified_name not in UNIFIED_LABEL_TO_ID:
                continue
            for color in colors or []:
                color_arr = np.asarray(color, dtype=np.uint8)
                if color_arr.shape != (3,):
                    continue
                out[np.all(rgb == color_arr, axis=-1)] = UNIFIED_LABEL_TO_ID[unified_name]
        return out

    if mask.ndim == 3:
        mask = mask[..., 0]

    mask_values = (config or {}).get("mask_label_values") or {
        "background": [0],
        "crop": [1],
        "weed": [2],
    }
    for unified_name, values in mask_values.items():
        unified_name = normalize_label_name(unified_name)
        if unified_name not in UNIFIED_LABEL_TO_ID:
            continue
        for value in values or []:
            out[mask == int(value)] = UNIFIED_LABEL_TO_ID[unified_name]
    return out
