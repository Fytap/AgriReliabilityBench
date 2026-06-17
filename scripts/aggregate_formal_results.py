from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any


DATASETS = ("cropandweed", "phenobench")
CLASSIFICATION_MODELS = ("convnextv2_tiny", "resnet50", "dinov2_small", "efficientnetv2_s", "swin_tiny")
SEGMENTATION_MODELS = ("deeplabv3plus", "unet", "segformer_b0", "upernet_swin_tiny", "mask2former_swin_tiny")
DETECTION_MODELS = ("yolov5l6u", "yolov12n", "rtdetr_l", "yolov8m", "yolo11m")

MODEL_LABELS = {
    "convnextv2_tiny": "ConvNeXtV2-Tiny",
    "resnet50": "ResNet50",
    "dinov2_small": "DINOv2-S/14 linear",
    "efficientnetv2_s": "EfficientNetV2-S",
    "swin_tiny": "Swin-T",
    "deeplabv3plus": "DeepLabV3+",
    "unet": "U-Net",
    "segformer_b0": "SegFormer-B0",
    "upernet_swin_tiny": "UPerNet-Swin-T",
    "mask2former_swin_tiny": "Mask2Former-Swin-T",
    "yolov5l6u": "YOLOv5l6u",
    "yolov8m": "YOLOv8m",
    "yolo11m": "YOLO11m",
    "yolov12n": "YOLOv12N",
    "rtdetr_l": "RT-DETR-L",
}


def main() -> None:
    root = Path(".").resolve()
    out_dir = root / "outputs" / "metrics"
    report_dir = root / "outputs" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    eval_rows = []
    eval_rows.extend(classification_rows(root))
    eval_rows.extend(segmentation_rows(root))
    eval_rows.extend(detection_rows(root))
    eval_rows = sorted(eval_rows, key=lambda r: (r["task"], r["model"], r["train_dataset"], r["test_dataset"], r["scope"]))

    deployment = deployment_lookup(root)
    reliability_rows = build_reliability_rows(eval_rows, deployment)
    training_rows = training_best_rows(root)

    write_csv(out_dir / "formal_eval_long.csv", eval_rows)
    write_csv(out_dir / "cross_dataset_reliability_summary.csv", reliability_rows)
    write_csv(out_dir / "paper_ready_reliability_deployment_summary.csv", reliability_rows)
    write_csv(out_dir / "classification_reliability_summary.csv", [r for r in reliability_rows if r["task"] == "classification"])
    write_csv(out_dir / "segmentation_reliability_summary.csv", [r for r in reliability_rows if r["task"] == "segmentation"])
    write_csv(out_dir / "detection_reliability_summary.csv", [r for r in reliability_rows if r["task"] == "detection"])
    write_csv(out_dir / "training_best_summary.csv", training_rows)
    write_report(report_dir / "formal_experiment_summary.md", eval_rows, reliability_rows, training_rows, root)

    print(f"wrote {out_dir / 'formal_eval_long.csv'} rows={len(eval_rows)}")
    print(f"wrote {out_dir / 'cross_dataset_reliability_summary.csv'} rows={len(reliability_rows)}")
    print(f"wrote {out_dir / 'training_best_summary.csv'} rows={len(training_rows)}")


def classification_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for path in sorted((root / "runs" / "indomain_eval" / "classification").glob("*_indomain_300ep.csv")):
        model, dataset = parse_model_dataset(path.stem.replace("_indomain_300ep", ""), CLASSIFICATION_MODELS)
        row = read_csv_one(path)
        rows.append(
            base_eval_row("classification", model, dataset, dataset, "in_domain", path)
            | {
                "primary_metric_name": "balanced_accuracy",
                "primary_metric": as_float(row.get("balanced_accuracy")),
                "accuracy": as_float(row.get("accuracy")),
                "balanced_accuracy": as_float(row.get("balanced_accuracy")),
                "macro_f1": as_float(row.get("macro_f1")),
                "ece": as_float(row.get("ece")),
                "brier": as_float(row.get("brier")),
                "high_conf_wrong_0.9": as_int(row.get("high_conf_wrong_0.9")),
                **classification_risk_metrics(path.with_name(path.stem + "_predictions.csv")),
            }
        )

    pattern = re.compile(r"(.+)_train_(cropandweed|phenobench)_test_(cropandweed|phenobench)_300ep$")
    for path in sorted((root / "runs" / "cross_eval" / "classification").glob("*_300ep.csv")):
        match = pattern.match(path.stem)
        if not match:
            continue
        model_raw, train_dataset, test_dataset = match.groups()
        model = normalize_model(model_raw, CLASSIFICATION_MODELS)
        row = read_csv_one(path)
        rows.append(
            base_eval_row("classification", model, train_dataset, test_dataset, "cross_dataset", path)
            | {
                "primary_metric_name": "balanced_accuracy",
                "primary_metric": as_float(row.get("balanced_accuracy")),
                "accuracy": as_float(row.get("accuracy")),
                "balanced_accuracy": as_float(row.get("balanced_accuracy")),
                "macro_f1": as_float(row.get("macro_f1")),
                "ece": as_float(row.get("ece")),
                "brier": as_float(row.get("brier")),
                "high_conf_wrong_0.9": as_int(row.get("high_conf_wrong_0.9")),
                **classification_risk_metrics(path.with_name(path.stem + "_predictions.csv")),
            }
        )
    return rows


def segmentation_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()

    for path in sorted((root / "runs" / "indomain_eval" / "segmentation").glob("*_indomain_300ep.csv")):
        model, dataset = parse_model_dataset(path.stem.replace("_indomain_300ep", ""), SEGMENTATION_MODELS)
        row = read_csv_one(path)
        rows.append(segmentation_eval_row(model, dataset, dataset, "in_domain", path, row))

    pattern = re.compile(r"(.+)_train_(cropandweed|phenobench)_test_(cropandweed|phenobench)_300ep$")
    paths = list((root / "runs" / "cross_eval" / "segmentation").glob("*_300ep.csv"))
    paths += list((root / "runs" / "cross_eval_final" / "segmentation").glob("*_300ep.csv"))
    for path in sorted(paths, key=lambda p: ("cross_eval_final" not in str(p), str(p))):
        match = pattern.match(path.stem)
        if not match:
            continue
        model_raw, train_dataset, test_dataset = match.groups()
        model = normalize_model(model_raw, SEGMENTATION_MODELS)
        key = (model, train_dataset, test_dataset, "cross_dataset")
        if key in seen:
            continue
        seen.add(key)
        row = read_csv_one(path)
        rows.append(segmentation_eval_row(model, train_dataset, test_dataset, "cross_dataset", path, row))
    return rows


def segmentation_eval_row(
    model: str,
    train_dataset: str,
    test_dataset: str,
    scope: str,
    path: Path,
    row: dict[str, str],
) -> dict[str, Any]:
    return base_eval_row("segmentation", model, train_dataset, test_dataset, scope, path) | {
        "primary_metric_name": "miou",
        "primary_metric": as_float(row.get("miou")),
        "miou": as_float(row.get("miou")),
        "iou_background": as_float(row.get("iou_background")),
        "iou_crop": as_float(row.get("iou_crop")),
        "iou_weed": as_float(row.get("iou_weed")),
        "loss": as_float(row.get("loss")),
    }


def detection_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for log_path in sorted((root / "runs" / "indomain_eval" / "detection").glob("*_indomain_300ep.log")):
        model, dataset = parse_model_dataset(log_path.stem.replace("_indomain_300ep", ""), DETECTION_MODELS)
        metrics = detection_metrics_for_log(log_path)
        rows.append(detection_eval_row(model, dataset, dataset, "in_domain", log_path, metrics))

    pattern = re.compile(r"(.+)_train_(cropandweed|phenobench)_test_(cropandweed|phenobench)_300ep$")
    for log_path in sorted((root / "runs" / "cross_eval_final" / "detection").glob("*_300ep.log")):
        match = pattern.match(log_path.stem)
        if not match:
            continue
        model_raw, train_dataset, test_dataset = match.groups()
        model = normalize_model(model_raw, DETECTION_MODELS)
        metrics = detection_metrics_for_log(log_path)
        rows.append(detection_eval_row(model, train_dataset, test_dataset, "cross_dataset", log_path, metrics))
    return rows


def detection_eval_row(
    model: str,
    train_dataset: str,
    test_dataset: str,
    scope: str,
    log_path: Path,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return base_eval_row("detection", model, train_dataset, test_dataset, scope, log_path) | {
        "primary_metric_name": "mAP50-95",
        "primary_metric": as_float(metrics.get("mAP50-95")),
        "precision": as_float(metrics.get("precision")),
        "recall": as_float(metrics.get("recall")),
        "mAP50": as_float(metrics.get("mAP50")),
        "mAP50-95": as_float(metrics.get("mAP50-95")),
    }


def base_eval_row(
    task: str,
    model: str,
    train_dataset: str,
    test_dataset: str,
    scope: str,
    source_path: Path,
) -> dict[str, Any]:
    return {
        "task": task,
        "model": model,
        "model_label": MODEL_LABELS.get(model, model),
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        "scope": scope,
        "source_file": str(source_path),
    }


def build_reliability_rows(eval_rows: list[dict[str, Any]], deployment: dict[tuple[str, str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    indomain = {
        (row["task"], row["model"], row["train_dataset"]): row
        for row in eval_rows
        if row["scope"] == "in_domain" and row["train_dataset"] == row["test_dataset"]
    }
    out = []
    for row in eval_rows:
        if row["scope"] != "cross_dataset":
            continue
        base = indomain.get((row["task"], row["model"], row["train_dataset"]))
        if not base:
            continue
        in_score = as_float(base.get("primary_metric"))
        external_score = as_float(row.get("primary_metric"))
        drop = in_score - external_score if is_number(in_score) and is_number(external_score) else ""
        retention = external_score / in_score if is_number(in_score) and in_score else ""
        reliability = {
            "task": row["task"],
            "model": row["model"],
            "model_label": row["model_label"],
            "train_dataset": row["train_dataset"],
            "external_test_dataset": row["test_dataset"],
            "primary_metric_name": row["primary_metric_name"],
            "in_domain_score": in_score,
            "external_score": external_score,
            "absolute_drop": drop,
            "retention": retention,
            "in_domain_source": base["source_file"],
            "external_source": row["source_file"],
        }
        deploy = deployment.get((row["task"], row["model"], row["train_dataset"]), {})
        for key in [
            "checkpoint_size_mb",
            "params",
            "params_millions",
            "batch_size",
            "img_size",
            "latency_ms",
            "fps",
            "peak_vram_mb",
            "backend",
            "architecture",
        ]:
            if key in deploy:
                reliability[f"deployment_{key}"] = deploy[key]
        for key in [
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "ece",
            "brier",
            "high_conf_wrong_0.9",
            "risk_coverage_auc",
            "risk_at_80_coverage",
            "risk_at_90_coverage",
            "miou",
            "iou_background",
            "iou_crop",
            "iou_weed",
            "precision",
            "recall",
            "mAP50",
            "mAP50-95",
        ]:
            if key in row:
                reliability[f"external_{key}"] = row.get(key)
            if key in base:
                reliability[f"in_domain_{key}"] = base.get(key)
        out.append(reliability)
    return sorted(out, key=lambda r: (r["task"], r["model"], r["train_dataset"], r["external_test_dataset"]))


def deployment_lookup(root: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    path = root / "outputs" / "metrics" / "deployment_metrics.csv"
    if not path.exists():
        return {}
    out = {}
    for row in read_csv_all(path):
        out[(row["task"], row["model"], row["dataset"])] = row
    return out


def training_best_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for dataset in DATASETS:
        for model in CLASSIFICATION_MODELS:
            path = classification_train_path(root, model, dataset)
            rows.append(training_best_row("classification", model, dataset, path, "balanced_accuracy"))
        for model in SEGMENTATION_MODELS:
            path = segmentation_train_path(root, model, dataset)
            rows.append(training_best_row("segmentation", model, dataset, path, "miou"))
        for model in DETECTION_MODELS:
            path = root / "runs" / "detection" / f"{model}_{dataset}_300ep" / "results.csv"
            rows.append(training_best_row("detection", model, dataset, path, "metrics/mAP50-95(B)"))
    return rows


def training_best_row(task: str, model: str, dataset: str, path: Path, score_col: str) -> dict[str, Any]:
    base = {
        "task": task,
        "model": model,
        "model_label": MODEL_LABELS.get(model, model),
        "dataset": dataset,
        "metric": score_col,
        "training_file": str(path),
        "last_epoch": "",
        "best_epoch": "",
        "best_score": "",
        "status": "missing",
    }
    if not path.exists():
        return base
    rows = read_csv_all(path)
    if not rows:
        return base | {"status": "empty"}
    best_epoch = ""
    best_score = -math.inf
    for idx, row in enumerate(rows, start=1):
        score = as_float(row.get(score_col))
        if is_number(score) and score > best_score:
            best_score = score
            best_epoch = row.get("epoch") or row.get("                  epoch") or idx
    last = rows[-1]
    return base | {
        "last_epoch": last.get("epoch") or last.get("                  epoch") or len(rows),
        "best_epoch": best_epoch,
        "best_score": best_score if best_score > -math.inf else "",
        "status": "ok",
    }


def classification_train_path(root: Path, model: str, dataset: str) -> Path:
    if model == "dinov2_small":
        return root / "runs" / "classification" / f"dinov2_small_linear_{dataset}_300ep" / "metrics.csv"
    if model in {"efficientnetv2_s", "swin_tiny"}:
        return root / "runs" / "classification" / f"{model}_{dataset}_300ep" / "metrics.csv"
    return root / "runs" / "classification" / f"{model}_{dataset}_paperparams" / "metrics.csv"


def segmentation_train_path(root: Path, model: str, dataset: str) -> Path:
    if model == "mask2former_swin_tiny" and dataset == "cropandweed":
        ddp_path = root / "runs" / "segmentation" / "mask2former_swin_tiny_cropandweed_300ep_ddp7_resume" / "metrics.csv"
        if ddp_path.exists():
            return ddp_path
    return root / "runs" / "segmentation" / f"{model}_{dataset}_300ep" / "metrics.csv"


def classification_risk_metrics(pred_path: Path) -> dict[str, Any]:
    if not pred_path.exists():
        return {}
    rows = read_csv_all(pred_path)
    if not rows:
        return {}
    pairs = []
    for row in rows:
        confidence = as_float(row.get("confidence"))
        correct = as_int(row.get("correct"))
        if is_number(confidence) and correct in (0, 1):
            pairs.append((float(confidence), int(correct)))
    if not pairs:
        return {}
    pairs.sort(key=lambda item: item[0], reverse=True)
    risks = []
    errors = 0
    n = len(pairs)
    risk_at = {0.8: "", 0.9: ""}
    prev_cov, prev_risk = 0.0, 0.0
    auc = 0.0
    for idx, (_, correct) in enumerate(pairs, start=1):
        errors += 1 - correct
        coverage = idx / n
        risk = errors / idx
        risks.append((coverage, risk))
        auc += (coverage - prev_cov) * (risk + prev_risk) / 2.0
        prev_cov, prev_risk = coverage, risk
    for target in risk_at:
        risk_at[target] = next((risk for coverage, risk in risks if coverage >= target), "")
    return {
        "risk_coverage_auc": auc,
        "risk_at_80_coverage": risk_at[0.8],
        "risk_at_90_coverage": risk_at[0.9],
    }


def detection_metrics_for_log(log_path: Path) -> dict[str, Any]:
    run_dir = log_path.with_suffix("")
    metrics_json = run_dir / "metrics.json"
    if metrics_json.exists():
        data = json.loads(metrics_json.read_text(encoding="utf-8"))
        return {
            "precision": data.get("metrics/precision(B)"),
            "recall": data.get("metrics/recall(B)"),
            "mAP50": data.get("metrics/mAP50(B)"),
            "mAP50-95": data.get("metrics/mAP50-95(B)"),
        }
    if not log_path.exists():
        return {}
    ansi = re.compile(r"\x1b\[[0-9;]*m")
    all_line = None
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        clean = ansi.sub("", line)
        if re.match(r"^\s*all\s+\d+", clean):
            all_line = clean
    if not all_line:
        return {}
    parts = all_line.split()
    if len(parts) < 7:
        return {}
    return {
        "precision": as_float(parts[-4]),
        "recall": as_float(parts[-3]),
        "mAP50": as_float(parts[-2]),
        "mAP50-95": as_float(parts[-1]),
    }


def parse_model_dataset(stem: str, models: tuple[str, ...]) -> tuple[str, str]:
    for dataset in DATASETS:
        suffix = "_" + dataset
        if stem.endswith(suffix):
            return normalize_model(stem[: -len(suffix)], models), dataset
    raise ValueError(f"Could not parse model/dataset from {stem}")


def normalize_model(model: str, models: tuple[str, ...]) -> str:
    if model in models:
        return model
    aliases = {
        "dinov2_small_linear": "dinov2_small",
        "vit_small_patch14_dinov2": "dinov2_small",
    }
    if model in aliases:
        return aliases[model]
    for candidate in sorted(models, key=len, reverse=True):
        if model.startswith(candidate):
            return candidate
    raise ValueError(f"Unknown model: {model}")


def read_csv_one(path: Path) -> dict[str, str]:
    rows = read_csv_all(path)
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows[0]


def read_csv_all(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, eval_rows: list[dict[str, Any]], reliability_rows: list[dict[str, Any]], training_rows: list[dict[str, Any]], root: Path) -> None:
    lines = [
        "# Formal Experiment Summary",
        "",
        "## Scope",
        "",
        "- Tasks: classification, detection, segmentation.",
        "- Datasets: CropAndWeed and PhenoBench.",
        "- Cross-dataset reliability is computed only within the same task and model.",
        "- Primary metrics: balanced accuracy for classification, mIoU for segmentation, mAP50-95 for detection.",
        "",
        "## Output Tables",
        "",
        "- `outputs/metrics/formal_eval_long.csv`: all in-domain and cross-dataset evaluations in one long table.",
        "- `outputs/metrics/cross_dataset_reliability_summary.csv`: in-domain vs external score, absolute drop, and retention.",
        "- `outputs/metrics/classification_reliability_summary.csv`: classification-only reliability rows.",
        "- `outputs/metrics/segmentation_reliability_summary.csv`: segmentation-only reliability rows.",
        "- `outputs/metrics/detection_reliability_summary.csv`: detection-only reliability rows.",
        "- `outputs/metrics/paper_ready_reliability_deployment_summary.csv`: cross-dataset reliability with deployment columns attached.",
        "- `outputs/metrics/training_best_summary.csv`: best epoch and best validation score from training logs.",
        "- `outputs/metrics/deployment_metrics.csv`: batch=1 latency, FPS, parameter count, checkpoint size, and peak VRAM.",
        "- `outputs/metrics/classification_stress_summary.csv`: classification corruption robustness summary.",
        "- `outputs/metrics/segmentation_stress_summary.csv`: segmentation corruption robustness summary.",
        "- `outputs/metrics/detection_stress_summary.csv`: detection corruption robustness summary.",
        "- `outputs/metrics/stress_reliability_summary.csv`: task-normalized clean-to-corruption drop and retention table.",
        "",
        "## Coverage",
        "",
        f"- Evaluation rows: {len(eval_rows)}.",
        f"- Cross-dataset reliability rows: {len(reliability_rows)}.",
        f"- Training summary rows: {len(training_rows)}.",
        "",
        "## Best In-Domain Scores",
        "",
    ]
    lines.extend(markdown_best_indomain(eval_rows))
    lines.extend(["", "## Cross-Dataset Retention Highlights", ""])
    lines.extend(markdown_retention(reliability_rows))
    stress = stress_highlights(root)
    if stress:
        lines.extend(["", "## Stress Robustness Highlights", ""])
        lines.extend(stress)
    deployment = deployment_highlights(root)
    if deployment:
        lines.extend(["", "## Deployment Highlights", ""])
        lines.extend(deployment)
    lines.extend(["", "## Notes", ""])
    lines.append("- RT-DETR evaluations were parsed from the dedicated `ultralytics.RTDETR` path because generic `yolo detect val` mis-read the saved RT-DETR outputs.")
    lines.append("- The tables intentionally do not compare metrics across tasks directly.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_best_indomain(eval_rows: list[dict[str, Any]]) -> list[str]:
    lines = []
    for task in ("classification", "segmentation", "detection"):
        for dataset in DATASETS:
            candidates = [
                row for row in eval_rows
                if row["task"] == task and row["scope"] == "in_domain" and row["test_dataset"] == dataset
            ]
            if not candidates:
                continue
            best = max(candidates, key=lambda row: as_float(row.get("primary_metric")) if is_number(as_float(row.get("primary_metric"))) else -math.inf)
            lines.append(
                f"- {task} / {dataset}: {best['model_label']} = {float(best['primary_metric']):.4f} "
                f"({best['primary_metric_name']})."
            )
    return lines


def markdown_retention(reliability_rows: list[dict[str, Any]]) -> list[str]:
    lines = []
    for task in ("classification", "segmentation", "detection"):
        rows = [row for row in reliability_rows if row["task"] == task and is_number(as_float(row.get("retention")))]
        if not rows:
            continue
        best = max(rows, key=lambda row: as_float(row["retention"]))
        worst = min(rows, key=lambda row: as_float(row["retention"]))
        lines.append(
            f"- {task}: best retention {best['model_label']} "
            f"{best['train_dataset']}→{best['external_test_dataset']} = {float(best['retention']):.3f}; "
            f"largest drop {worst['model_label']} {worst['train_dataset']}→{worst['external_test_dataset']} "
            f"drop = {float(worst['absolute_drop']):.4f}."
        )
    return lines


def stress_highlights(root: Path) -> list[str]:
    path = root / "outputs" / "metrics" / "stress_reliability_summary.csv"
    if not path.exists():
        return []
    rows = [row for row in read_csv_all(path) if row.get("corruption") != "clean"]
    if not rows:
        return []
    out = []
    for task in ("classification", "segmentation", "detection"):
        task_rows = [
            row for row in rows
            if row.get("task") == task and is_number(as_float(row.get("primary_metric_absolute_drop")))
        ]
        if not task_rows:
            continue
        worst = max(task_rows, key=lambda row: as_float(row["primary_metric_absolute_drop"]))
        best = min(task_rows, key=lambda row: as_float(row["primary_metric_absolute_drop"]))
        out.append(
            f"- {task}: largest {worst['primary_metric_name']} stress drop "
            f"{worst['run_id']} under {worst['corruption']} = {float(worst['primary_metric_absolute_drop']):.4f}; "
            f"smallest drop {best['run_id']} under {best['corruption']} = {float(best['primary_metric_absolute_drop']):.4f}."
        )
    return out


def deployment_highlights(root: Path) -> list[str]:
    path = root / "outputs" / "metrics" / "deployment_metrics.csv"
    if not path.exists():
        return []
    rows = read_csv_all(path)
    out = []
    for task in ("classification", "segmentation", "detection"):
        task_rows = [row for row in rows if row.get("task") == task and is_number(as_float(row.get("latency_ms")))]
        if not task_rows:
            continue
        fastest = min(task_rows, key=lambda row: as_float(row["latency_ms"]))
        lightest = min(task_rows, key=lambda row: as_float(row["params_millions"]))
        out.append(
            f"- {task}: fastest {MODEL_LABELS.get(fastest['model'], fastest['model'])} "
            f"({fastest['dataset']}) = {float(fastest['latency_ms']):.2f} ms/image; "
            f"smallest {MODEL_LABELS.get(lightest['model'], lightest['model'])} = {float(lightest['params_millions']):.2f}M params."
        )
    return out


def as_float(value: Any) -> float:
    if value in (None, ""):
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def as_int(value: Any) -> int | str:
    if value in (None, ""):
        return ""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return ""


def is_number(value: Any) -> bool:
    return isinstance(value, (float, int)) and math.isfinite(float(value))


if __name__ == "__main__":
    main()
