#!/usr/bin/env python
"""Bootstrap segmentation IoU metrics from matched prediction masks and GT masks.

Input schema:
  --manifest CSV with image/sample rows and mask_path column for ground truth
  --prediction-dir directory containing prediction masks named <sample_id>.png
  --repo-root optional root used to resolve relative mask_path entries
  --out-csv output table with mIoU/background/crop/weed IoU mean and CI

Prediction masks must use the unified ids: 0 background, 1 crop, 2 weed.
"""

from __future__ import annotations

import argparse, csv, random
from pathlib import Path

import numpy as np
from PIL import Image


def load_mask(path):
    return np.asarray(Image.open(path))


def per_image_ious(pred, gt):
    vals = []
    for cls in [0, 1, 2]:
        p = pred == cls
        g = gt == cls
        union = np.logical_or(p, g).sum()
        inter = np.logical_and(p, g).sum()
        vals.append(float(inter / union) if union else float("nan"))
    miou = float(np.nanmean(vals))
    return {"mIoU": miou, "background_IoU": vals[0], "crop_IoU": vals[1], "weed_IoU": vals[2]}


def summarize(rows, metric, reps, seed):
    random.seed(seed)
    vals = [r[metric] for r in rows if not np.isnan(r[metric])]
    if not vals:
        return np.nan, np.nan, np.nan
    boots = []
    for _ in range(reps):
        sample = [random.choice(vals) for _ in vals]
        boots.append(float(np.mean(sample)))
    return float(np.mean(vals)), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--prediction-dir", required=True)
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--replicates", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    root = Path(args.repo_root)
    pred_dir = Path(args.prediction_dir)
    rows = []
    with open(args.manifest, newline="", encoding="utf-8") as f:
        for rec in csv.DictReader(f):
            sid = rec.get("sample_id") or Path(rec["image_path"]).stem
            pred_path = pred_dir / f"{sid}.png"
            gt_path = Path(rec["mask_path"])
            if not gt_path.is_absolute():
                gt_path = root / gt_path
            if pred_path.exists() and gt_path.exists():
                rows.append(per_image_ious(load_mask(pred_path), load_mask(gt_path)))

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "estimate", "ci_low", "ci_high", "n_images", "n_bootstrap"])
        for metric in ["mIoU", "background_IoU", "crop_IoU", "weed_IoU"]:
            mean, lo, hi = summarize(rows, metric, args.replicates, args.seed)
            w.writerow([metric, mean, lo, hi, len(rows), args.replicates])


if __name__ == "__main__":
    main()
