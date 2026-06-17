from __future__ import annotations

import argparse
import csv
import json
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter
import timm
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from eval_classification import classification_metrics, predict


CORRUPTIONS = (
    "clean",
    "gaussian_blur_3",
    "low_resolution_0.5",
    "jpeg_quality_50",
    "brightness_minus_20",
    "contrast_0.7",
)


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
    parser.add_argument("--corruption", required=True, choices=CORRUPTIONS)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ds = StressWeedPresenceDataset(args.csv, repo_root, args.split, args.img_size, args.corruption)
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
        "corruption": args.corruption,
        "num_samples": len(ds),
        **metrics,
    }
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    out_path.with_suffix(".json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    print(row, flush=True)


class StressWeedPresenceDataset(Dataset):
    def __init__(self, csv_path: str, repo_root: Path, split: str, img_size: int, corruption: str) -> None:
        self.rows = []
        with Path(csv_path).open("r", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["split"] == split:
                    self.rows.append(row)
        self.repo_root = repo_root
        self.corruption = corruption
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
        image = apply_corruption(image, self.corruption)
        return self.transform(image), int(row["label"])


def apply_corruption(image: Image.Image, corruption: str) -> Image.Image:
    if corruption == "clean":
        return image
    if corruption == "gaussian_blur_3":
        return image.filter(ImageFilter.GaussianBlur(radius=3))
    if corruption == "low_resolution_0.5":
        width, height = image.size
        small = image.resize((max(1, width // 2), max(1, height // 2)), Image.Resampling.BILINEAR)
        return small.resize((width, height), Image.Resampling.BILINEAR)
    if corruption == "jpeg_quality_50":
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=50)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")
    if corruption == "brightness_minus_20":
        return ImageEnhance.Brightness(image).enhance(0.8)
    if corruption == "contrast_0.7":
        return ImageEnhance.Contrast(image).enhance(0.7)
    raise ValueError(f"Unsupported corruption: {corruption}")


if __name__ == "__main__":
    main()
