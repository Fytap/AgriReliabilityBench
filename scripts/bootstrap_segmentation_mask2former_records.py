from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from transformers import Mask2FormerForUniversalSegmentation

from train_segmentation_mask2former import SegDataset, collate_batch, semantic_predictions


def class_ious(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3) -> list[float]:
    vals = []
    for cls in range(n_classes):
        gt = y_true == cls
        pr = y_pred == cls
        union = np.logical_or(gt, pr).sum()
        inter = np.logical_and(gt, pr).sum()
        vals.append(float(inter / union) if union else float("nan"))
    return vals


def summarize(values: list[float], replicates: int, rng: random.Random) -> tuple[float, float, float]:
    arr = np.asarray([v for v in values if not np.isnan(v)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    boots = []
    for _ in range(replicates):
        idx = [rng.randrange(arr.size) for _ in range(arr.size)]
        boots.append(float(arr[idx].mean()))
    return float(arr.mean()), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


@torch.no_grad()
def evaluate_run(run: dict[str, str], repo_root: Path, out_dir: Path, batch_size: int, img_size: int) -> list[dict]:
    ds = SegDataset(run["csv"], repo_root, "val", img_size, run["mask_format"])
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_batch,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        "facebook/mask2former-swin-tiny-ade-semantic",
        num_labels=3,
        ignore_mismatched_sizes=True,
    )
    checkpoint = torch.load(repo_root / run["checkpoint"], map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    rows = []
    offset = 0
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        masks = batch["semantic_masks"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            outputs = model(pixel_values=pixel_values)
        preds = semantic_predictions(outputs.class_queries_logits, outputs.masks_queries_logits, masks.shape[-2:])
        masks_np = masks.cpu().numpy()
        preds_np = preds.cpu().numpy()
        for i in range(preds_np.shape[0]):
            dataset_row = ds.rows[offset + i]
            ious = class_ious(masks_np[i], preds_np[i])
            rows.append(
                {
                    "run_id": run["run_id"],
                    "model": run["model"],
                    "train_dataset": run["train_dataset"],
                    "test_dataset": run["test_dataset"],
                    "scope": run["scope"],
                    "sample_id": dataset_row.get("sample_id", Path(dataset_row["image_path"]).stem),
                    "mIoU": float(np.nanmean(ious)),
                    "background_IoU": ious[0],
                    "crop_IoU": ious[1],
                    "weed_IoU": ious[2],
                }
            )
        offset += preds_np.shape[0]

    out_dir.mkdir(parents=True, exist_ok=True)
    per_image_path = out_dir / f"{run['run_id']}_per_image.csv"
    with per_image_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-csv", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--per-image-dir", required=True)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=18)
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    out_rows = []
    rng = random.Random(args.seed)
    with open(args.runs_csv, newline="", encoding="utf-8") as f:
        runs = list(csv.DictReader(f))

    for run in runs:
        rows = evaluate_run(run, repo_root, Path(args.per_image_dir), args.batch_size, args.img_size)
        for metric in ["mIoU", "background_IoU", "crop_IoU", "weed_IoU"]:
            estimate, lo, hi = summarize([float(r[metric]) for r in rows], args.replicates, rng)
            out_rows.append(
                {
                    "task": "segmentation",
                    "model": run["model"],
                    "train_dataset": run["train_dataset"],
                    "test_dataset": run["test_dataset"],
                    "scope": run["scope"],
                    "metric": metric,
                    "estimate": estimate,
                    "ci_low": lo,
                    "ci_high": hi,
                    "n_images": len(rows),
                    "n_bootstrap": args.replicates,
                    "status": "available_image_resampling",
                    "method_note": "per-image IoU bootstrap from rerun Mask2Former predictions and ground-truth masks",
                    "run_id": run["run_id"],
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
