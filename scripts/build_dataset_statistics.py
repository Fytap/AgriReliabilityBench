from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(".").resolve()
OUT_DIR = ROOT / "outputs" / "metrics"
MANUSCRIPT = ROOT / "outputs" / "manuscript"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MANUSCRIPT.mkdir(parents=True, exist_ok=True)

    rows = []
    for dataset in ("cropandweed", "phenobench"):
        rows.extend(classification_stats(dataset))
        rows.extend(segmentation_stats(dataset))
        rows.extend(detection_stats(dataset))

    write_csv(OUT_DIR / "dataset_task_statistics.csv", rows)
    (MANUSCRIPT / "10_dataset_statistics_methods.md").write_text(to_markdown(rows), encoding="utf-8")
    print(f"wrote {OUT_DIR / 'dataset_task_statistics.csv'} rows={len(rows)}")
    print(f"wrote {MANUSCRIPT / '10_dataset_statistics_methods.md'}")


def classification_stats(dataset: str) -> list[dict[str, object]]:
    path = ROOT / "data" / "converted" / dataset / "classification" / "weed_presence.csv"
    rows = read_rows(path)
    out = []
    for split, group in group_by_split(rows).items():
        labels = Counter(row.get("label", "") for row in group)
        out.append(
            base(dataset, "classification", split)
            | {
                "samples": len(group),
                "images": len({row.get("image_path", "") for row in group}),
                "masks": "",
                "label_files": "",
                "objects": "",
                "weed_positive": labels.get("1", 0),
                "weed_negative": labels.get("0", 0),
            }
        )
    return out


def segmentation_stats(dataset: str) -> list[dict[str, object]]:
    path = ROOT / "data" / "converted" / dataset / "segmentation" / "semantic_segmentation.csv"
    rows = read_rows(path)
    out = []
    for split, group in group_by_split(rows).items():
        out.append(
            base(dataset, "segmentation", split)
            | {
                "samples": len(group),
                "images": len({row.get("image_path", "") for row in group}),
                "masks": len({row.get("mask_path", "") for row in group if row.get("mask_path")}),
                "label_files": "",
                "objects": "",
                "weed_positive": "",
                "weed_negative": "",
            }
        )
    return out


def detection_stats(dataset: str) -> list[dict[str, object]]:
    root = ROOT / "data" / "converted" / dataset / "yolo_detection"
    out = []
    for split in ("train", "val", "test"):
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        image_count = count_files(image_dir, {".jpg", ".jpeg", ".png", ".bmp"})
        label_paths = list_files(label_dir, {".txt"})
        objects = 0
        nonempty = 0
        for path in label_paths:
            try:
                n = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
            except UnicodeDecodeError:
                n = sum(1 for line in path.read_text(errors="ignore").splitlines() if line.strip())
            objects += n
            if n:
                nonempty += 1
        if image_count == 0 and not label_paths:
            continue
        out.append(
            base(dataset, "detection", split)
            | {
                "samples": image_count,
                "images": image_count,
                "masks": "",
                "label_files": len(label_paths),
                "objects": objects,
                "weed_positive": nonempty,
                "weed_negative": max(image_count - nonempty, 0),
            }
        )
    return out


def base(dataset: str, task: str, split: str) -> dict[str, object]:
    return {
        "dataset": dataset,
        "task": task,
        "split": split,
    }


def group_by_split(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("split", "unknown")].append(row)
    return dict(sorted(grouped.items()))


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def list_files(path: Path, suffixes: set[str]) -> list[Path]:
    if not path.exists():
        return []
    return [child for child in path.iterdir() if child.is_file() and child.suffix.lower() in suffixes]


def count_files(path: Path, suffixes: set[str]) -> int:
    return len(list_files(path, suffixes))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def to_markdown(rows: list[dict[str, object]]) -> str:
    lines = [
        "# Dataset Task Statistics for Methods",
        "",
        "These counts are computed from the converted benchmark files and should be used in the Materials and Methods section.",
        "",
        "| dataset | task | split | samples | images | masks | label_files | objects | weed_positive | weed_negative |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {task} | {split} | {samples} | {images} | {masks} | {label_files} | {objects} | {weed_positive} | {weed_negative} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "Suggested Methods sentence:",
            "",
            "The converted benchmark statistics are reported by dataset, task, and split in Table S1. Classification labels were derived as weed-present versus weed-absent, segmentation uses semantic crop/weed/background masks, and detection labels follow YOLO-format crop/weed boxes generated from available object annotations or connected mask regions.",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
