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
from torchvision.transforms import functional as TF
import segmentation_models_pytorch as smp


CROPANDWEED_REMAP = {2: 0, 0: 1, 1: 2}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--arch", required=True, choices=["deeplabv3plus", "unet", "segformer", "upernet"])
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=672)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--mask-format", default="cropandweed", choices=["cropandweed", "unified", "phenobench"])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ds = SegDataset(args.csv, repo_root, args.split, args.img_size, args.mask_format)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.arch, args.encoder)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)

    metrics = evaluate(model, loader, nn.CrossEntropyLoss(), device)
    row = {
        "checkpoint": args.checkpoint,
        "csv": args.csv,
        "split": args.split,
        "arch": args.arch,
        "encoder": args.encoder,
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


def build_model(arch: str, encoder: str):
    if arch == "deeplabv3plus":
        return smp.DeepLabV3Plus(encoder_name=encoder, encoder_weights=None, in_channels=3, classes=3)
    if arch == "segformer":
        return smp.Segformer(encoder_name=encoder, encoder_weights=None, in_channels=3, classes=3)
    if arch == "upernet":
        return smp.UPerNet(encoder_name=encoder, encoder_weights=None, in_channels=3, classes=3)
    return smp.Unet(encoder_name=encoder, encoder_weights=None, in_channels=3, classes=3)


class SegDataset(Dataset):
    def __init__(self, csv_path: str, repo_root: Path, split: str, img_size: int, mask_format: str) -> None:
        self.rows = []
        with Path(csv_path).open("r", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["split"] == split:
                    self.rows.append(row)
        self.repo_root = repo_root
        self.img_size = img_size
        self.mask_format = mask_format

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = Image.open(self.repo_root / row["image_path"]).convert("RGB")
        mask = Image.open(self.repo_root / row["mask_path"])
        image = TF.resize(image, [self.img_size, self.img_size], interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [self.img_size, self.img_size], interpolation=TF.InterpolationMode.NEAREST)
        image_t = TF.to_tensor(image)
        image_t = TF.normalize(image_t, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        return image_t, torch.from_numpy(remap_mask(np.asarray(mask, dtype=np.int64), self.mask_format))


def remap_mask(mask_arr: np.ndarray, mask_format: str) -> np.ndarray:
    if mask_format == "unified":
        out = np.asarray(mask_arr, dtype=np.int64).copy()
        out[(out < 0) | (out > 2)] = 0
        return out

    out = np.zeros_like(mask_arr, dtype=np.int64)
    if mask_format == "cropandweed":
        for src, dst in CROPANDWEED_REMAP.items():
            out[mask_arr == src] = dst
        return out

    out[mask_arr == 0] = 0
    out[(mask_arr == 1) | (mask_arr == 3)] = 1
    out[(mask_arr == 2) | (mask_arr == 4)] = 2
    return out


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> dict[str, float]:
    model.eval()
    losses = []
    confusion = np.zeros((3, 3), dtype=np.float64)
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, masks)
        preds = logits.argmax(dim=1)
        losses.append(float(loss.detach().cpu()))
        confusion += confusion_matrix(masks.cpu().numpy(), preds.cpu().numpy(), 3)

    intersection = np.diag(confusion)
    union = confusion.sum(axis=1) + confusion.sum(axis=0) - intersection
    iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "miou": float(np.mean(iou)),
        "iou_background": float(iou[0]),
        "iou_crop": float(iou[1]),
        "iou_weed": float(iou[2]),
    }


def confusion_matrix(y_true, y_pred, num_classes: int) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.int64).ravel()
    mask = (y_true >= 0) & (y_true < num_classes)
    return np.bincount(num_classes * y_true[mask] + y_pred[mask], minlength=num_classes**2).reshape(num_classes, num_classes)


if __name__ == "__main__":
    main()
