from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agri_reliability.data.label_mapping import mask_to_unified
from agri_reliability.utils.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-root", default="data/converted/cropandweed")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--dataset-config")
    parser.add_argument("--preserve-splits", action="store_true")
    parser.add_argument("--seed", type=int, default=20260523)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    out_root = (repo_root / args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    dataset_config = load_yaml(args.dataset_config) if args.dataset_config else None

    records = list(read_jsonl(repo_root / args.manifest))
    rows = []
    for rec in records:
        split = rec.get("split") if args.preserve_splits else deterministic_split(rec["sample_id"], args.seed)
        if not split:
            split = deterministic_split(rec["sample_id"], args.seed)
        rows.append({**rec, "split": split})

    write_classification(rows, out_root, repo_root)
    write_segmentation(rows, out_root, repo_root, dataset_config)
    write_yolo_detection(rows, out_root, repo_root)
    print(f"Prepared {len(rows)} records under {out_root}")


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def deterministic_split(sample_id: str, seed: int) -> str:
    value = int(hashlib.md5(f"{seed}:{sample_id}".encode("utf-8")).hexdigest()[:8], 16)
    ratio = value / 0xFFFFFFFF
    if ratio < 0.70:
        return "train"
    if ratio < 0.85:
        return "val"
    return "test"


def write_classification(rows: list[dict], out_root: Path, repo_root: Path) -> None:
    out_dir = out_root / "classification"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "weed_presence.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "split", "image_path", "label"])
        writer.writeheader()
        for rec in rows:
            image_label = rec.get("image_label") or {}
            if "weed_presence" not in image_label:
                continue
            writer.writerow(
                {
                    "sample_id": rec["sample_id"],
                    "split": rec["split"],
                    "image_path": rec["image_path"],
                    "label": int(image_label["weed_presence"]),
                }
            )


def write_segmentation(rows: list[dict], out_root: Path, repo_root: Path, dataset_config: dict | None) -> None:
    out_dir = out_root / "segmentation"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "semantic_segmentation.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "split", "image_path", "mask_path"],
        )
        writer.writeheader()
        for rec in rows:
            if not rec.get("mask_path"):
                continue
            mask_path = rec["mask_path"]
            if dataset_config:
                raw_mask_path = repo_root / rec["mask_path"]
                unified_dir = out_dir / "masks" / rec["split"]
                unified_dir.mkdir(parents=True, exist_ok=True)
                mask_path = str(unified_dir / f"{rec['sample_id']}.png")
                unified = mask_to_unified(np.asarray(Image.open(raw_mask_path)), dataset_config)
                Image.fromarray(unified.astype(np.uint8)).save(mask_path)
            writer.writerow(
                {
                    "sample_id": rec["sample_id"],
                    "split": rec["split"],
                    "image_path": rec["image_path"],
                    "mask_path": mask_path,
                }
            )


def write_yolo_detection(rows: list[dict], out_root: Path, repo_root: Path) -> None:
    out_dir = out_root / "yolo_detection"
    for split in ("train", "val", "test"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    for rec in rows:
        split = rec["split"]
        image_path = repo_root / rec["image_path"]
        if not image_path.exists():
            continue
        label_path = out_dir / "labels" / split / f"{image_path.stem}.txt"
        image_link = out_dir / "images" / split / image_path.name
        safe_symlink(image_path, image_link)

        boxes = read_cropandweed_boxes(repo_root / rec["boxes_path"]) if rec.get("boxes_path") else []
        boxes.extend(rec.get("boxes") or [])
        with label_path.open("w", encoding="utf-8") as handle:
            width, height = Image.open(image_path).size
            for box in boxes:
                parsed = parse_box(box)
                if parsed is None:
                    continue
                cls, x1, y1, x2, y2, inclusive_max = parsed
                x_center = ((x1 + x2) / 2.0) / width
                y_center = ((y1 + y2) / 2.0) / height
                extent = 1.0 if inclusive_max else 0.0
                box_width = max(0.0, (x2 - x1 + extent) / width)
                box_height = max(0.0, (y2 - y1 + extent) / height)
                handle.write(f"{cls} {x_center:.8f} {y_center:.8f} {box_width:.8f} {box_height:.8f}\n")

    dataset_name = str(rows[0].get("dataset", out_root.name)).lower() if rows else out_root.name
    yaml_path = out_dir / f"{dataset_name}.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {out_dir}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "names:",
                "  0: crop",
                "  1: weed",
                "",
            ]
        ),
        encoding="utf-8",
    )


def parse_box(box) -> tuple[int, float, float, float, float, bool] | None:
    if isinstance(box, dict):
        label = str(box.get("label", "")).lower()
        label_id = box.get("label_id")
        if label == "crop" or label_id == 1:
            cls = 0
        elif label == "weed" or label_id == 2:
            cls = 1
        else:
            return None
        x1 = float(box.get("x_min", box.get("xmin", box.get("x1", 0))))
        y1 = float(box.get("y_min", box.get("ymin", box.get("y1", 0))))
        x2 = float(box.get("x_max", box.get("xmax", box.get("x2", 0))))
        y2 = float(box.get("y_max", box.get("ymax", box.get("y2", 0))))
        return cls, x1, y1, x2, y2, True

    cls, x1, y1, x2, y2 = box
    cls = int(cls)
    if cls not in (0, 1):
        return None
    return cls, float(x1), float(y1), float(x2), float(y2), False


def read_cropandweed_boxes(path: Path) -> list[tuple[int, float, float, float, float]]:
    boxes = []
    if not path.exists():
        return boxes
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = [part.strip() for part in line.strip().split(",")]
            if len(parts) < 5:
                continue
            x1, y1, x2, y2 = [float(value) for value in parts[:4]]
            cls = int(float(parts[4]))
            boxes.append((cls, x1, y1, x2, y2))
    return boxes


def safe_symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    rel_src = os.path.relpath(src, dst.parent)
    os.symlink(rel_src, dst)


if __name__ == "__main__":
    main()
