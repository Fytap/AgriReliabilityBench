from __future__ import annotations

import argparse
import csv
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
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--model", default="convnextv2_tiny.fcmae_ft_in1k")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--freeze-backbone", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = WeedPresenceDataset(args.csv, repo_root, "train", args.img_size)
    val_ds = WeedPresenceDataset(args.csv, repo_root, "val", args.img_size)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(args.model, pretrained=True, num_classes=2)
    if args.freeze_backbone:
        freeze_for_linear_probe(model)
    model.to(device)

    weights = class_weights(train_ds.labels).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=max(1, args.warmup_epochs),
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, args.epochs - args.warmup_epochs),
            ),
        ],
        milestones=[max(1, args.warmup_epochs)],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_balanced = -1.0
    log_path = out_dir / "metrics.csv"
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "train_loss", "val_loss", "accuracy", "balanced_accuracy"],
        )
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
            val_loss, metrics = evaluate(model, val_loader, criterion, device)
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                **metrics,
            }
            writer.writerow(row)
            handle.flush()
            print(row, flush=True)
            torch.save({"model": model.state_dict(), "epoch": epoch, "args": vars(args)}, out_dir / "last.pt")
            if metrics["balanced_accuracy"] > best_balanced:
                best_balanced = metrics["balanced_accuracy"]
                torch.save({"model": model.state_dict(), "epoch": epoch, "args": vars(args)}, out_dir / "best.pt")
            scheduler.step()


class WeedPresenceDataset(Dataset):
    def __init__(self, csv_path: str, repo_root: Path, split: str, img_size: int) -> None:
        self.rows = []
        with Path(csv_path).open("r", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["split"] == split:
                    self.rows.append(row)
        self.repo_root = repo_root
        self.labels = [int(row["label"]) for row in self.rows]
        if split == "train":
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(img_size, scale=(0.55, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )
        else:
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
        label = int(row["label"])
        return self.transform(image), label


def class_weights(labels: list[int]) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=2).astype(float)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32)


def freeze_for_linear_probe(model: nn.Module) -> None:
    head_tokens = ("head", "fc", "classifier")
    trainable = 0
    for name, param in model.named_parameters():
        param.requires_grad = any(token in name for token in head_tokens)
        trainable += int(param.requires_grad)
    if trainable == 0:
        raise ValueError("Could not identify a classifier head to train for linear probing.")


def train_one_epoch(model, loader, criterion, optimizer, scaler, device) -> float:
    model.train()
    losses = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    losses, labels_all, preds_all = [], [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels)
        preds = logits.argmax(dim=1)
        losses.append(float(loss.detach().cpu()))
        labels_all.extend(labels.cpu().numpy().tolist())
        preds_all.extend(preds.cpu().numpy().tolist())
    labels_arr = np.asarray(labels_all)
    preds_arr = np.asarray(preds_all)
    accuracy = float(np.mean(labels_arr == preds_arr))
    recalls = []
    for cls in (0, 1):
        mask = labels_arr == cls
        recalls.append(float(np.mean(preds_arr[mask] == cls)) if np.any(mask) else 0.0)
    return float(np.mean(losses)), {"accuracy": accuracy, "balanced_accuracy": float(np.mean(recalls))}


if __name__ == "__main__":
    main()
