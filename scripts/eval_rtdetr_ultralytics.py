from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import RTDETR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--save-json", action="store_true")
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args()

    model = RTDETR(args.model)
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        save_json=args.save_json,
        plots=args.plots,
    )
    save_dir = Path(args.project) / args.name
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = save_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics.results_dict, indent=2), encoding="utf-8")
    print(metrics.results_dict, flush=True)


if __name__ == "__main__":
    main()
