from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from train_segmentation_mask2former import SegDataset, collate_batch, confusion_matrix, semantic_predictions
from transformers import Mask2FormerForUniversalSegmentation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--model-name", default="facebook/mask2former-swin-tiny-ade-semantic")
    parser.add_argument("--split", default="val")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--mask-format", default="cropandweed", choices=["cropandweed", "unified", "phenobench"])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ds = SegDataset(args.csv, repo_root, args.split, args.img_size, args.mask_format)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_batch,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        args.model_name,
        num_labels=3,
        ignore_mismatched_sizes=True,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    metrics = evaluate(model, loader, device)

    row = {
        "checkpoint": args.checkpoint,
        "csv": args.csv,
        "split": args.split,
        "model_name": args.model_name,
        "mask_format": args.mask_format,
        "num_samples": len(ds),
        **metrics,
    }
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    out_path.with_suffix(".json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    print(row, flush=True)


@torch.no_grad()
def evaluate(model, loader, device) -> dict[str, float]:
    model.eval()
    confusion = np.zeros((3, 3), dtype=np.float64)
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        masks = batch["semantic_masks"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            outputs = model(pixel_values=pixel_values)
        preds = semantic_predictions(outputs.class_queries_logits, outputs.masks_queries_logits, masks.shape[-2:])
        confusion += confusion_matrix(masks.cpu().numpy(), preds.cpu().numpy(), 3)
    intersection = np.diag(confusion)
    union = confusion.sum(axis=1) + confusion.sum(axis=0) - intersection
    iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
    return {
        "loss": float("nan"),
        "miou": float(np.mean(iou)),
        "iou_background": float(iou[0]),
        "iou_crop": float(iou[1]),
        "iou_weed": float(iou[2]),
    }


if __name__ == "__main__":
    main()
