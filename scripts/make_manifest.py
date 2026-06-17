from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from agri_reliability.data.manifest import ManifestRecord, write_jsonl
from agri_reliability.data.cropandweed_adapter import build_cropandweed_manifest
from agri_reliability.data.phenobench_adapter import build_phenobench_manifest
from agri_reliability.utils.config import load_yaml


def scan_image_folder(root: Path, dataset: str, split: str, task: str):
    records = []
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    for label_idx, class_dir in enumerate(sorted([p for p in root.iterdir() if p.is_dir()])):
        label_name = class_dir.name
        for img in class_dir.rglob('*'):
            if img.suffix.lower() in exts:
                records.append(ManifestRecord(
                    sample_id=f'{dataset}_{split}_{len(records):08d}',
                    dataset=dataset,
                    split=split,
                    task=task,
                    image_path=str(img),
                    label=label_idx,
                    label_name=label_name,
                    modality='rgb',
                    metadata={},
                ))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Dataset YAML config for CropAndWeed or PhenoBench')
    parser.add_argument('--root')
    parser.add_argument('--dataset')
    parser.add_argument('--split', default='train')
    parser.add_argument('--task', default='classification')
    parser.add_argument('--out', required=True)
    parser.add_argument('--max-records', type=int)
    parser.add_argument('--no-derive', action='store_true')
    args = parser.parse_args()

    if args.config:
        config = load_yaml(args.config)
        builder = {
            'cropandweed': build_cropandweed_manifest,
            'phenobench': build_phenobench_manifest,
        }.get(config['name'])
        if builder is None:
            raise SystemExit(f"Unsupported configured dataset: {config['name']}")
        result = builder(
            config,
            max_records=args.max_records,
            derive_from_masks=not args.no_derive,
        )
        records = result.records
        for warning in result.warnings:
            print(f"WARNING: {warning}")
    else:
        if not args.root or not args.dataset:
            raise SystemExit('Provide either --config or both --root and --dataset.')
        records = scan_image_folder(Path(args.root), args.dataset, args.split, args.task)

    write_jsonl(records, args.out)
    print(f'Wrote {len(records)} records to {args.out}')


if __name__ == '__main__':
    main()
