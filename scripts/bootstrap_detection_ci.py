#!/usr/bin/env python
"""Bootstrap detection metrics from COCO-like predictions and YOLO labels.

Input schema:
  --predictions JSON list with image_id, category_id, bbox [x,y,w,h], score
  --labels-dir directory with YOLO txt labels named <image_id>.txt
  --images-dir directory with images used to recover width/height for YOLO labels
  --out-csv output table with metric, estimate, ci_low, ci_high, n_images

This script intentionally requires matched predictions, labels, and images. It
does not infer confidence intervals from aggregate logs.
"""

from __future__ import annotations

import argparse, csv, json, random
from collections import defaultdict
from pathlib import Path
from PIL import Image

import numpy as np


def xywh_iou(a, b):
    ax1, ay1, aw, ah = a; bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def load_gt(labels_dir, images_dir):
    gt = defaultdict(list)
    for label_path in Path(labels_dir).glob("*.txt"):
        image_id = label_path.stem
        img = next(Path(images_dir).glob(image_id + ".*"), None)
        if img is None:
            continue
        w, h = Image.open(img).size
        for line in label_path.read_text().splitlines():
            if not line.strip():
                continue
            cls, xc, yc, bw, bh = map(float, line.split()[:5])
            x = (xc - bw / 2) * w
            y = (yc - bh / 2) * h
            gt[image_id].append({"category_id": int(cls), "bbox": [x, y, bw * w, bh * h]})
    return dict(gt)


def compute(preds, gt, image_ids, iou_thr):
    tp = fp = fn = 0
    scores = []
    labels = []
    for image_id in image_ids:
        gts = list(gt.get(image_id, []))
        matched = set()
        ps = sorted([p for p in preds.get(image_id, [])], key=lambda x: -x.get("score", 0))
        for p in ps:
            best_iou, best_j = 0.0, None
            for j, g in enumerate(gts):
                if j in matched or int(g["category_id"]) != int(p["category_id"]):
                    continue
                iou = xywh_iou(p["bbox"], g["bbox"])
                if iou > best_iou:
                    best_iou, best_j = iou, j
            ok = best_iou >= iou_thr and best_j is not None
            if ok:
                tp += 1; matched.add(best_j); labels.append(1)
            else:
                fp += 1; labels.append(0)
            scores.append(float(p.get("score", 0)))
        fn += max(0, len(gts) - len(matched))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    if not scores:
        ap = 0.0
    else:
        order = np.argsort(-np.asarray(scores))
        y = np.asarray(labels)[order]
        cum_tp = np.cumsum(y)
        cum_fp = np.cumsum(1 - y)
        total_gt = sum(len(gt.get(i, [])) for i in image_ids)
        rec = cum_tp / max(total_gt, 1)
        prec = cum_tp / np.maximum(cum_tp + cum_fp, 1)
        ap = 0.0
        for t in np.linspace(0, 1, 101):
            vals = prec[rec >= t]
            ap += (vals.max() if vals.size else 0.0) / 101.0
    return {"precision": precision, "recall": recall, "AP": ap}


def summarize(vals):
    arr = np.asarray(vals, dtype=float)
    return float(arr.mean()), float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--labels-dir", required=True)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--replicates", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)

    pred_list = json.loads(Path(args.predictions).read_text())
    preds = defaultdict(list)
    for p in pred_list:
        preds[str(p["image_id"])].append(p)
    gt = load_gt(args.labels_dir, args.images_dir)
    image_ids = sorted(set(gt) | set(preds))
    thresholds = [0.5] + [round(x, 2) for x in np.arange(0.5, 0.96, 0.05)]

    boot = {k: [] for k in ["precision", "recall", "mAP50", "mAP50-95"]}
    for _ in range(args.replicates):
        sample = [random.choice(image_ids) for _ in image_ids]
        m50 = compute(preds, gt, sample, 0.5)
        aps = [compute(preds, gt, sample, t)["AP"] for t in thresholds[1:]]
        boot["precision"].append(m50["precision"])
        boot["recall"].append(m50["recall"])
        boot["mAP50"].append(m50["AP"])
        boot["mAP50-95"].append(float(np.mean(aps)))

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "estimate", "ci_low", "ci_high", "n_images", "n_bootstrap"])
        for metric, vals in boot.items():
            mean, lo, hi = summarize(vals)
            w.writerow([metric, mean, lo, hi, len(image_ids), args.replicates])


if __name__ == "__main__":
    main()
