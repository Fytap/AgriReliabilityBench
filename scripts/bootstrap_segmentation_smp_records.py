from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from eval_segmentation_smp import SegDataset, build_model


METRICS = ["miou", "iou_background", "iou_crop", "iou_weed"]


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 3) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.int64).ravel()
    mask = (y_true >= 0) & (y_true < num_classes)
    return np.bincount(num_classes * y_true[mask] + y_pred[mask], minlength=num_classes**2).reshape(
        num_classes, num_classes
    )


def ious_from_confusion(confusion: np.ndarray) -> dict[str, float]:
    intersection = np.diag(confusion)
    union = confusion.sum(axis=1) + confusion.sum(axis=0) - intersection
    iou = np.divide(intersection, union, out=np.full_like(intersection, np.nan, dtype=np.float64), where=union > 0)
    return {
        "miou": float(np.nanmean(iou)),
        "iou_background": float(iou[0]),
        "iou_crop": float(iou[1]),
        "iou_weed": float(iou[2]),
    }


@torch.no_grad()
def evaluate_run(run: dict[str, str], repo_root: Path, per_image_dir: Path) -> list[dict[str, object]]:
    img_size = int(run["img_size"])
    batch_size = int(run.get("batch_size", "4") or 4)
    ds = SegDataset(run["csv"], repo_root, "val", img_size, run["mask_format"])
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(run["arch"], run["encoder"])
    checkpoint = torch.load(repo_root / run["checkpoint"], map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    rows: list[dict[str, object]] = []
    offset = 0
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(images)
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        targets = masks.numpy()
        for i in range(targets.shape[0]):
            sample_row = ds.rows[offset + i]
            cm = confusion_matrix(targets[i], preds[i], 3)
            metrics = ious_from_confusion(cm)
            rows.append(
                {
                    "run_id": run["run_id"],
                    "model": run["model"],
                    "train_dataset": run["train_dataset"],
                    "test_dataset": run["test_dataset"],
                    "scope": run["scope"],
                    "image_path": sample_row.get("image_path", ""),
                    "mask_path": sample_row.get("mask_path", ""),
                    **metrics,
                }
            )
        offset += targets.shape[0]

    per_image_dir.mkdir(parents=True, exist_ok=True)
    per_path = per_image_dir / f"{run['run_id']}_per_image.csv"
    with per_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def bootstrap(values: np.ndarray, reps: int, rng: np.random.Generator) -> tuple[float, float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    idx = rng.integers(0, values.size, size=(reps, values.size))
    samples = values[idx].mean(axis=1)
    return float(values.mean()), float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-csv", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--per-image-dir", required=True)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=18)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    per_image_dir = Path(args.per_image_dir).resolve()
    rng = np.random.default_rng(args.seed)

    summary_rows: list[dict[str, object]] = []
    with Path(args.runs_csv).open("r", encoding="utf-8") as handle:
        for run in csv.DictReader(handle):
            try:
                rows = evaluate_run(run, repo_root, per_image_dir)
                for metric in METRICS:
                    values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
                    estimate, ci_low, ci_high = bootstrap(values, args.replicates, rng)
                    summary_rows.append(
                        {
                            "task": "segmentation",
                            "model": run["model"],
                            "train_dataset": run["train_dataset"],
                            "test_dataset": run["test_dataset"],
                            "scope": run["scope"],
                            "metric": metric,
                            "estimate": estimate,
                            "ci_low": ci_low,
                            "ci_high": ci_high,
                            "n_images": len(rows),
                            "n_bootstrap": args.replicates,
                            "status": "available",
                            "method_note": "Per-image semantic-segmentation IoU bootstrap from saved SMP checkpoints and validation masks.",
                            "run_id": run["run_id"],
                        }
                    )
            except Exception as exc:  # keep the audit complete without inventing unavailable rows
                for metric in METRICS:
                    summary_rows.append(
                        {
                            "task": "segmentation",
                            "model": run.get("model", ""),
                            "train_dataset": run.get("train_dataset", ""),
                            "test_dataset": run.get("test_dataset", ""),
                            "scope": run.get("scope", ""),
                            "metric": metric,
                            "estimate": "",
                            "ci_low": "",
                            "ci_high": "",
                            "n_images": "",
                            "n_bootstrap": args.replicates,
                            "status": "unavailable",
                            "method_note": str(exc).replace("\n", " ")[:500],
                            "run_id": run.get("run_id", ""),
                        }
                    )

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)


if __name__ == "__main__":
    main()
