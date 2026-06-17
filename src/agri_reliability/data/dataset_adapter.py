from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

from agri_reliability.data.derive_tasks import derive_fields_from_mask_path
from agri_reliability.data.label_mapping import unified_label_space
from agri_reliability.data.manifest import ManifestRecord


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MASK_EXTS = IMAGE_EXTS
BOX_EXTS = {".json", ".csv", ".txt"}
MASK_HINTS = ("mask", "masks", "label", "labels", "semantic", "semantics", "seg", "gt")
BOX_HINTS = ("box", "boxes", "bbox", "bboxes", "detect", "detection")
IMAGE_HINTS = ("image", "images", "rgb", "left", "camera")
SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "test",
    "testing": "test",
}


@dataclass
class AdapterResult:
    records: list[ManifestRecord]
    warnings: list[str]


def build_manifest_from_config(
    config: dict,
    *,
    task: str = "multi_task",
    max_records: int | None = None,
    derive_from_masks: bool = True,
) -> AdapterResult:
    dataset = config["name"]
    root = Path(config.get("root", ""))
    warnings: list[str] = []
    if not root.exists():
        return AdapterResult([], [f"{dataset}: dataset root not found: {root}"])

    image_files = _find_image_files(root, config)
    if not image_files:
        return AdapterResult([], [f"{dataset}: no image files found under {root}"])
    if max_records is not None:
        image_files = image_files[:max_records]

    mask_index = _index_sidecar_files(root, config, "mask_dirs", MASK_HINTS, MASK_EXTS)
    box_index = _index_sidecar_files(root, config, "boxes_dirs", BOX_HINTS, BOX_EXTS)
    records: list[ManifestRecord] = []

    for idx, image_path in enumerate(image_files):
        key = _match_key(image_path)
        mask_path = mask_index.get(key)
        boxes_path = box_index.get(key)
        image_label = None
        boxes = None
        metadata = {
            "manifest_version": 1,
            "has_mask": bool(mask_path),
            "has_boxes_path": bool(boxes_path),
        }

        if derive_from_masks and mask_path:
            try:
                image_label, boxes = derive_fields_from_mask_path(mask_path, config)
                metadata["derived_from_mask"] = True
                metadata["derived_box_count"] = len(boxes)
            except Exception as exc:
                metadata["derived_from_mask"] = False
                metadata["derive_error"] = str(exc)
                warnings.append(f"{dataset}: failed to derive labels from {mask_path}: {exc}")

        record = ManifestRecord(
            sample_id=f"{dataset}_{_infer_split(image_path, config)}_{idx:08d}",
            dataset=dataset,
            split=_infer_split(image_path, config),
            task=task,
            image_path=str(image_path),
            image_label=image_label,
            boxes_path=str(boxes_path) if boxes_path else None,
            boxes=boxes,
            bbox_path=str(boxes_path) if boxes_path else None,
            mask_path=str(mask_path) if mask_path else None,
            unified_label_space=unified_label_space(),
            modality=config.get("modality", "rgb"),
            metadata=metadata,
        )
        records.append(record)

    return AdapterResult(records, warnings)


def _find_image_files(root: Path, config: dict) -> list[Path]:
    configured_dirs = _existing_dirs(root, config.get("image_dirs") or [])
    if configured_dirs:
        candidates = _files_with_exts(configured_dirs, IMAGE_EXTS)
    else:
        candidates = _files_with_exts([root], IMAGE_EXTS)
    return sorted([path for path in candidates if _looks_like_image(path)])


def _index_sidecar_files(
    root: Path,
    config: dict,
    config_key: str,
    path_hints: tuple[str, ...],
    exts: set[str],
) -> dict[str, Path]:
    configured_dirs = _existing_dirs(root, config.get(config_key) or [])
    if configured_dirs:
        index: dict[str, Path] = {}
        for directory in configured_dirs:
            for path in sorted(_files_with_exts([directory], exts)):
                key = _match_key(path)
                index.setdefault(key, path)
        return index
    else:
        candidates = [
            path
            for path in _files_with_exts([root], exts)
            if _path_contains_any(path, path_hints)
        ]
    index: dict[str, Path] = {}
    for path in sorted(candidates):
        key = _match_key(path)
        index.setdefault(key, path)
    return index


def _files_with_exts(roots: list[Path], exts: set[str]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in exts:
                files.append(path)
    return files


def _existing_dirs(root: Path, dirs: list[str]) -> list[Path]:
    found = []
    for item in dirs:
        path = Path(item)
        if not path.is_absolute():
            path = root / path
        if path.exists() and path.is_dir():
            found.append(path)
    return found


def _looks_like_image(path: Path) -> bool:
    lower_parts = [part.lower() for part in path.parts]
    if any(hint in part for part in lower_parts for hint in MASK_HINTS + BOX_HINTS):
        return any(hint in part for part in lower_parts for hint in IMAGE_HINTS)
    return True


def _path_contains_any(path: Path, hints: tuple[str, ...]) -> bool:
    return any(hint in part.lower() for part in path.parts for hint in hints)


def _match_key(path: Path) -> str:
    stem = path.stem.lower()
    suffixes = (
        "_mask",
        "_masks",
        "_label",
        "_labels",
        "_semantic",
        "_semantics",
        "_seg",
        "_gt",
        "_rgb",
        "_image",
        "_boxes",
        "_bbox",
    )
    changed = True
    while changed:
        changed = False
        for suffix in suffixes:
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                changed = True
    return stem


def _infer_split(path: Path, config: dict) -> str:
    split_rules = config.get("split_names") or SPLIT_ALIASES
    normalized_rules = {str(key).lower(): str(value) for key, value in split_rules.items()}
    for part in path.parts:
        split = normalized_rules.get(part.lower())
        if split:
            return split
    return str(config.get("default_split", "unknown"))


def read_boxes_json(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as file_handle:
        payload = json.load(file_handle)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("boxes", "annotations", "objects"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    return []
