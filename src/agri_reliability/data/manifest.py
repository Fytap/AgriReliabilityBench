from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json


@dataclass
class ManifestRecord:
    sample_id: str
    dataset: str
    split: str
    task: str
    image_path: str
    image_label: dict | None = None
    label: int | None = None
    label_name: str | None = None
    boxes_path: str | None = None
    boxes: list[dict] | None = None
    bbox_path: str | None = None
    mask_path: str | None = None
    unified_label_space: list[str] | None = None
    modality: str | None = None
    metadata: dict | None = None
    source_session: str | None = None


def write_jsonl(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for rec in records:
            if hasattr(rec, '__dataclass_fields__'):
                rec = asdict(rec)
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')


def read_jsonl(path):
    with Path(path).open('r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)
