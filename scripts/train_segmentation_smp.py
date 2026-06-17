from __future__ import annotations

import argparse
import csv
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
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--arch", default="deeplabv3plus", choices=["deeplabv3plus", "unet", "segformer", "upernet"])
    parser.add_argument("--encoder", default="resnet50")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--mask-format", default="cropandweed", choices=["cropandweed", "unified", "phenobench"])
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = SegDataset(args.csv, repo_root, "train", args.img_size, args.mask_format)
    val_ds = SegDataset(args.csv, repo_root, "val", args.img_size, args.mask_format)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.arch == "deeplabv3plus":
        model = smp.DeepLabV3Plus(encoder_name=args.encoder, encoder_weights="imagenet", in_channels=3, classes=3)
    elif args.arch == "segformer":
        model = smp.Segformer(encoder_name=args.encoder, encoder_weights="imagenet", in_channels=3, classes=3)
    elif args.arch == "upernet":
        model = smp.UPerNet(encoder_name=args.encoder, encoder_weights="imagenet", in_channels=3, classes=3)
    else:
        model = smp.Unet(encoder_name=args.encoder, encoder_weights="imagenet", in_channels=3, classes=3)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_miou = -1.0
    log_path = out_dir / "metrics.csv"
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_loss", "miou", "iou_background", "iou_crop", "iou_weed"])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
            val_loss, metrics = evaluate(model, val_loader, criterion, device)
            row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **metrics}
            writer.writerow(row)
            handle.flush()
            print(row, flush=True)
            torch.save({"model": model.state_dict(), "epoch": epoch, "args": vars(args)}, out_dir / "last.pt")
            if metrics["miou"] > best_miou:
                best_miou = metrics["miou"]
                torch.save({"model": model.state_dict(), "epoch": epoch, "args": vars(args)}, out_dir / "best.pt")


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
        mask_arr = np.asarray(mask, dtype=np.int64)
        out = remap_mask(mask_arr, self.mask_format)
        return image_t, torch.from_numpy(out)


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


def train_one_epoch(model, loader, criterion, optimizer, scaler, device) -> float:
    model.train()
    losses = []
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


@torch.no_grad()
def evaluate(model, loader, criterion, device):
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
    metrics = {
        "miou": float(np.mean(iou)),
        "iou_background": float(iou[0]),
        "iou_crop": float(iou[1]),
        "iou_weed": float(iou[2]),
    }
    return float(np.mean(losses)), metrics


def confusion_matrix(y_true, y_pred, num_classes: int) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.int64).ravel()
    mask = (y_true >= 0) & (y_true < num_classes)
    return np.bincount(num_classes * y_true[mask] + y_pred[mask], minlength=num_classes**2).reshape(num_classes, num_classes)


if __name__ == "__main__":
    main()
