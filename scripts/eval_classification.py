from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ds = WeedPresenceDataset(args.csv, repo_root, args.split, args.img_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(args.model, pretrained=False, num_classes=2)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)

    labels, preds, confidences, probs = predict(model, loader, device)
    metrics = classification_metrics(labels, preds, confidences, probs)
    row = {
        "checkpoint": args.checkpoint,
        "csv": args.csv,
        "split": args.split,
        "model": args.model,
        "num_samples": len(ds),
        **metrics,
    }
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    out_path.with_suffix(".json").write_text(json.dumps(row, indent=2), encoding="utf-8")

    pred_path = out_path.with_name(out_path.stem + "_predictions.csv")
    with pred_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["label", "pred", "confidence", "correct", "prob_0", "prob_1"])
        writer.writeheader()
        for label, pred, confidence, prob in zip(labels, preds, confidences, probs):
            writer.writerow(
                {
                    "label": int(label),
                    "pred": int(pred),
                    "confidence": float(confidence),
                    "correct": int(label == pred),
                    "prob_0": float(prob[0]),
                    "prob_1": float(prob[1]),
                }
            )
    print(row, flush=True)


class WeedPresenceDataset(Dataset):
    def __init__(self, csv_path: str, repo_root: Path, split: str, img_size: int) -> None:
        self.rows = []
        with Path(csv_path).open("r", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["split"] == split:
                    self.rows.append(row)
        self.repo_root = repo_root
        self.transform = transforms.Compose(
            [
                transforms.Resize(int(img_size * 1.15)),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = Image.open(self.repo_root / row["image_path"]).convert("RGB")
        return self.transform(image), int(row["label"])


@torch.no_grad()
def predict(model: nn.Module, loader, device):
    model.eval()
    labels_all, preds_all, conf_all, probs_all = [], [], [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
        conf, preds = probs.max(dim=1)
        labels_all.extend(labels.numpy().tolist())
        preds_all.extend(preds.cpu().numpy().tolist())
        conf_all.extend(conf.cpu().numpy().tolist())
        probs_all.extend(probs.cpu().numpy().tolist())
    return (
        np.asarray(labels_all, dtype=np.int64),
        np.asarray(preds_all, dtype=np.int64),
        np.asarray(conf_all, dtype=np.float64),
        np.asarray(probs_all, dtype=np.float64),
    )


def classification_metrics(labels: np.ndarray, preds: np.ndarray, confidences: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    correct = labels == preds
    recalls, precisions, f1s = [], [], []
    for cls in (0, 1):
        true_mask = labels == cls
        pred_mask = preds == cls
        tp = float(np.sum(true_mask & pred_mask))
        precision = tp / max(float(np.sum(pred_mask)), 1.0)
        recall = tp / max(float(np.sum(true_mask)), 1.0)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        recalls.append(recall)
        precisions.append(precision)
        f1s.append(f1)
    one_hot = np.eye(2, dtype=np.float64)[labels]
    return {
        "accuracy": float(np.mean(correct)),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "ece": expected_calibration_error(confidences, correct.astype(np.int64)),
        "brier": float(np.mean(np.sum((probs - one_hot) ** 2, axis=1))),
        "high_conf_wrong_0.9": int(np.sum((confidences >= 0.9) & ~correct)),
    }


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for idx in range(n_bins):
        lo, hi = bins[idx], bins[idx + 1]
        mask = (confidence > lo) & (confidence <= hi) if idx else (confidence >= lo) & (confidence <= hi)
        if not np.any(mask):
            continue
        acc = np.mean(correct[mask])
        conf = np.mean(confidence[mask])
        ece += np.mean(mask) * abs(acc - conf)
    return float(ece)


if __name__ == "__main__":
    main()
