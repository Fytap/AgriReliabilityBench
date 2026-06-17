from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


IOU_THRESHOLDS = [round(x, 2) for x in np.arange(0.50, 0.96, 0.05)]


def xywh_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def read_gt(labels_dir: Path, images_dir: Path) -> dict[str, list[dict]]:
    gt: dict[str, list[dict]] = {}
    for label_path in sorted(labels_dir.glob("*.txt")):
        image_id = label_path.stem
        image_path = None
        for ext in (".jpg", ".jpeg", ".png", ".JPG", ".PNG"):
            cand = images_dir / f"{image_id}{ext}"
            if cand.exists():
                image_path = cand
                break
        if image_path is None:
            continue
        width, height = Image.open(image_path).size
        boxes = []
        for line in label_path.read_text().splitlines():
            if not line.strip():
                continue
            cls, xc, yc, bw, bh = [float(x) for x in line.split()[:5]]
            boxes.append(
                {
                    "category_id": int(cls),
                    "bbox": [
                        (xc - bw / 2.0) * width,
                        (yc - bh / 2.0) * height,
                        bw * width,
                        bh * height,
                    ],
                }
            )
        gt[image_id] = boxes
    return gt


def read_preds(path: Path) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    rows = json.loads(path.read_text())
    for pred in rows:
        image_id = str(pred["image_id"])
        # Ultralytics JSON export in this project used 1/2 category ids while
        # YOLO labels use 0/1. Shift only when category ids are positive.
        cat = int(pred["category_id"])
        cat = cat - 1 if cat > 0 else cat
        grouped[image_id].append(
            {
                "category_id": cat,
                "bbox": [float(x) for x in pred["bbox"]],
                "score": float(pred.get("score", 0.0)),
            }
        )
    for image_id in grouped:
        grouped[image_id].sort(key=lambda x: -x["score"])
    return dict(grouped)


def ap_for_image(preds: list[dict], gt: list[dict], iou_thr: float) -> tuple[float, float, float]:
    if not gt and not preds:
        return 1.0, 1.0, 1.0
    if not gt:
        return 0.0, 0.0, 0.0
    matched: set[int] = set()
    tp_flags = []
    fp_flags = []
    for pred in preds:
        best_iou = 0.0
        best_j = None
        for j, target in enumerate(gt):
            if j in matched or int(target["category_id"]) != int(pred["category_id"]):
                continue
            iou = xywh_iou(pred["bbox"], target["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j is not None and best_iou >= iou_thr:
            matched.add(best_j)
            tp_flags.append(1.0)
            fp_flags.append(0.0)
        else:
            tp_flags.append(0.0)
            fp_flags.append(1.0)
    if not tp_flags:
        return 0.0, 0.0, 0.0
    tp = np.cumsum(np.asarray(tp_flags))
    fp = np.cumsum(np.asarray(fp_flags))
    recalls = tp / max(len(gt), 1)
    precisions = tp / np.maximum(tp + fp, 1)
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        vals = precisions[recalls >= t]
        ap += (float(vals.max()) if vals.size else 0.0) / 101.0
    precision = float(tp[-1] / max(tp[-1] + fp[-1], 1))
    recall = float(tp[-1] / max(len(gt), 1))
    return ap, precision, recall


def per_image_metrics(preds_by_image: dict[str, list[dict]], gt_by_image: dict[str, list[dict]]) -> list[dict[str, float]]:
    image_ids = sorted(set(gt_by_image) | set(preds_by_image))
    rows = []
    for image_id in image_ids:
        preds = preds_by_image.get(image_id, [])
        gt = gt_by_image.get(image_id, [])
        ap50, precision, recall = ap_for_image(preds, gt, 0.5)
        aps = [ap_for_image(preds, gt, thr)[0] for thr in IOU_THRESHOLDS]
        rows.append(
            {
                "image_id": image_id,
                "mAP50": ap50,
                "mAP50-95": float(np.mean(aps)),
                "precision": precision,
                "recall": recall,
            }
        )
    return rows


def ci(values: list[float], replicates: int, rng: random.Random) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    boots = []
    n = arr.size
    for _ in range(replicates):
        idx = [rng.randrange(n) for _ in range(n)]
        boots.append(float(arr[idx].mean()))
    return float(arr.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-csv", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--per-image-dir", required=True)
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    out_rows = []
    per_image_dir = Path(args.per_image_dir)
    per_image_dir.mkdir(parents=True, exist_ok=True)

    with open(args.runs_csv, newline="", encoding="utf-8") as f:
        runs = list(csv.DictReader(f))
    rng = random.Random(args.seed)

    for run in runs:
        target = run["test_dataset"]
        labels_dir = repo_root / "data" / "converted" / target / "yolo_detection" / "labels" / "val"
        images_dir = repo_root / "data" / "converted" / target / "yolo_detection" / "images" / "val"
        preds_path = repo_root / run["predictions_json"]
        gt = read_gt(labels_dir, images_dir)
        preds = read_preds(preds_path)
        rows = per_image_metrics(preds, gt)
        per_image_path = per_image_dir / f"{run['run_id']}_per_image.csv"
        with per_image_path.open("w", newline="", encoding="utf-8") as pf:
            writer = csv.DictWriter(pf, fieldnames=["image_id", "mAP50", "mAP50-95", "precision", "recall"])
            writer.writeheader()
            writer.writerows(rows)
        for metric in ["mAP50", "mAP50-95", "precision", "recall"]:
            estimate, lo, hi = ci([float(r[metric]) for r in rows], args.replicates, rng)
            out_rows.append(
                {
                    **{k: run[k] for k in ["run_id", "model", "train_dataset", "test_dataset", "scope"]},
                    "task": "detection",
                    "metric": metric,
                    "estimate": estimate,
                    "ci_low": lo,
                    "ci_high": hi,
                    "n_images": len(rows),
                    "n_bootstrap": args.replicates,
                    "status": "available_image_resampling",
                    "method_note": "per-image AP/precision/recall bootstrap from saved predictions.json and YOLO label files",
                }
            )

    fieldnames = [
        "task",
        "model",
        "train_dataset",
        "test_dataset",
        "scope",
        "metric",
        "estimate",
        "ci_low",
        "ci_high",
        "n_images",
        "n_bootstrap",
        "status",
        "method_note",
        "run_id",
    ]
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)


if __name__ == "__main__":
    main()
