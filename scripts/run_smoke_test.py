from __future__ import annotations

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd

from agri_reliability.data.cropandweed_adapter import build_cropandweed_manifest
from agri_reliability.data.manifest import write_jsonl
from agri_reliability.data.phenobench_adapter import build_phenobench_manifest
from agri_reliability.metrics.calibration import (
    error_detection_auroc,
    expected_calibration_error,
    maximum_calibration_error,
    multiclass_brier_score,
)
from agri_reliability.metrics.classification import classification_summary
from agri_reliability.metrics.reliability import (
    cross_dataset_summary,
    high_confidence_wrong_rate,
    risk_at_coverage,
)
from agri_reliability.metrics.risk_coverage import risk_coverage_auc, risk_coverage_curve
from agri_reliability.metrics.segmentation import segmentation_summary
from agri_reliability.reporting.plots import plot_reliability_diagram, plot_risk_coverage
from agri_reliability.utils.config import load_yaml


def main() -> None:
    rng = np.random.default_rng(20260522)
    out_dirs = _ensure_output_dirs()

    configs = {
        "cropandweed": load_yaml(ROOT / "configs/datasets/cropandweed.yaml"),
        "phenobench": load_yaml(ROOT / "configs/datasets/phenobench.yaml"),
    }
    (out_dirs["reports"] / "smoke_config.json").write_text(
        json.dumps(configs, indent=2),
        encoding="utf-8",
    )

    manifest_results = _build_manifests(configs, out_dirs["manifests"])
    dataset_summary = _write_dataset_summary(manifest_results, out_dirs["metrics"])
    _write_manifest_report(manifest_results, dataset_summary, out_dirs["reports"])

    classification_payload = _fake_classification_payload(rng)
    class_metrics = _classification_metrics(classification_payload)
    segmentation_metrics = _segmentation_metrics(rng)
    cross_dataset = _cross_dataset_rows(class_metrics)
    reliability_rows = _reliability_rows(class_metrics, cross_dataset)

    pd.DataFrame([class_metrics, segmentation_metrics]).to_csv(
        out_dirs["metrics"] / "smoke_metrics.csv",
        index=False,
    )
    pd.DataFrame(cross_dataset).to_csv(
        out_dirs["metrics"] / "cross_dataset_template.csv",
        index=False,
    )
    pd.DataFrame(reliability_rows).to_csv(
        out_dirs["metrics"] / "reliability_metrics_template.csv",
        index=False,
    )

    coverage, risk = risk_coverage_curve(
        classification_payload["confidence"],
        classification_payload["correct"],
    )
    pd.DataFrame({"coverage": coverage, "risk": risk}).to_csv(
        out_dirs["metrics"] / "risk_coverage.csv",
        index=False,
    )
    plot_reliability_diagram(
        classification_payload["confidence"],
        classification_payload["correct"],
        out_dirs["figures"] / "reliability_diagram.pdf",
        title="Smoke reliability diagram",
    )
    plot_risk_coverage(
        coverage,
        risk,
        out_dirs["figures"] / "risk_coverage_curve.pdf",
        title="Smoke risk-coverage curve",
    )
    _write_smoke_report(
        manifest_results,
        class_metrics,
        segmentation_metrics,
        cross_dataset,
        out_dirs["reports"],
    )
    print("Smoke test complete. See outputs/reports/smoke_report.md")


def _ensure_output_dirs() -> dict[str, Path]:
    out_dirs = {
        "manifests": ROOT / "outputs/manifests",
        "metrics": ROOT / "outputs/metrics",
        "figures": ROOT / "outputs/figures",
        "reports": ROOT / "outputs/reports",
    }
    for path in out_dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return out_dirs


def _build_manifests(configs: dict[str, dict], out_dir: Path) -> dict[str, dict]:
    builders = {
        "cropandweed": build_cropandweed_manifest,
        "phenobench": build_phenobench_manifest,
    }
    results: dict[str, dict] = {}
    for dataset, config in configs.items():
        result = builders[dataset](config, max_records=16, derive_from_masks=True)
        out_path = out_dir / f"{dataset}_manifest.jsonl"
        write_jsonl(result.records, out_path)
        results[dataset] = {
            "config": config,
            "records": result.records,
            "warnings": result.warnings,
            "out_path": out_path,
        }
    return results


def _write_dataset_summary(manifest_results: dict[str, dict], out_dir: Path) -> pd.DataFrame:
    rows = []
    for dataset, result in manifest_results.items():
        records = result["records"]
        rows.append(
            {
                "dataset": dataset,
                "root": result["config"].get("root"),
                "records": len(records),
                "splits": ",".join(sorted({rec.split for rec in records})) if records else "",
                "records_with_masks": sum(bool(rec.mask_path) for rec in records),
                "records_with_boxes": sum(bool(rec.boxes_path or rec.boxes) for rec in records),
                "records_with_image_label": sum(bool(rec.image_label) for rec in records),
                "status": "ok" if records else "skipped",
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "dataset_summary.csv", index=False)
    return df


def _write_manifest_report(
    manifest_results: dict[str, dict],
    dataset_summary: pd.DataFrame,
    out_dir: Path,
) -> None:
    lines = ["# Manifest Report", ""]
    lines.append("This report is generated by `scripts/run_smoke_test.py`.")
    lines.append("")
    lines.append("## Dataset Summary")
    lines.append("")
    lines.append(_df_to_markdown(dataset_summary))
    lines.append("")
    lines.append("## Warnings")
    lines.append("")
    any_warning = False
    for dataset, result in manifest_results.items():
        for warning in result["warnings"]:
            any_warning = True
            lines.append(f"- {warning}")
        if not result["records"] and not result["warnings"]:
            any_warning = True
            lines.append(f"- {dataset}: no records generated.")
    if not any_warning:
        lines.append("- None.")
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    for dataset, result in manifest_results.items():
        lines.append(f"- `{result['out_path'].relative_to(ROOT)}`")
    (out_dir / "manifest_report.md").write_text("\n".join(lines), encoding="utf-8")


def _fake_classification_payload(rng: np.random.Generator) -> dict[str, np.ndarray]:
    n = 96
    y_true = rng.integers(0, 2, size=n)
    logits = rng.normal(loc=0.0, scale=1.0, size=(n, 2))
    logits[np.arange(n), y_true] += 1.1
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    y_pred = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    correct = (y_true == y_pred).astype(int)
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "probs": probs,
        "confidence": confidence,
        "correct": correct,
    }


def _classification_metrics(payload: dict[str, np.ndarray]) -> dict[str, float | str]:
    metrics = classification_summary(payload["y_true"], payload["y_pred"])
    metrics.update(
        {
            "task": "classification",
            "dataset": "synthetic_cropandweed_phenobench_smoke",
            "model": "fake_reproducible_predictions",
            "ece": expected_calibration_error(payload["confidence"], payload["correct"]),
            "mce": maximum_calibration_error(payload["confidence"], payload["correct"]),
            "brier": multiclass_brier_score(payload["probs"], payload["y_true"]),
            "error_detection_auroc": error_detection_auroc(
                payload["confidence"],
                payload["correct"],
            ),
            "risk_coverage_auc": risk_coverage_auc(
                payload["confidence"],
                payload["correct"],
            ),
            "risk_at_80_coverage": risk_at_coverage(
                payload["confidence"],
                payload["correct"],
                0.80,
            ),
            "risk_at_90_coverage": risk_at_coverage(
                payload["confidence"],
                payload["correct"],
                0.90,
            ),
            "high_conf_wrong_rate": high_confidence_wrong_rate(
                payload["confidence"],
                payload["correct"],
                threshold=0.90,
            ),
            "metric_source": "synthetic_predictions",
        }
    )
    return metrics


def _segmentation_metrics(rng: np.random.Generator) -> dict[str, float | str]:
    y_true = np.zeros((10, 32, 32), dtype=np.uint8)
    y_pred = np.zeros_like(y_true)
    for idx in range(y_true.shape[0]):
        y_true[idx, 6:18, 5:15] = 1
        y_true[idx, 16:27, 18:29] = 2
        y_pred[idx] = y_true[idx]
        noise = rng.random((32, 32))
        y_pred[idx][noise < 0.05] = rng.integers(0, 3, size=int(np.sum(noise < 0.05)))
        if idx % 4 == 0:
            y_pred[idx, 16:27, 18:29] = 0

    metrics = segmentation_summary(y_true, y_pred, num_classes=3)
    metrics.update(
        {
            "task": "segmentation",
            "dataset": "synthetic_cropandweed_phenobench_smoke",
            "model": "fake_reproducible_predictions",
            "accuracy": np.nan,
            "macro_f1": np.nan,
            "balanced_accuracy": np.nan,
            "ece": np.nan,
            "mce": np.nan,
            "brier": np.nan,
            "error_detection_auroc": np.nan,
            "risk_coverage_auc": np.nan,
            "risk_at_80_coverage": np.nan,
            "risk_at_90_coverage": np.nan,
            "high_conf_wrong_rate": np.nan,
            "metric_source": "synthetic_masks",
        }
    )
    return metrics


def _cross_dataset_rows(class_metrics: dict[str, float | str]) -> list[dict[str, float | str]]:
    base_score = float(class_metrics["balanced_accuracy"])
    crop_in = min(0.98, base_score + 0.04)
    pheno_in = max(0.50, base_score - 0.02)
    experiments = [
        ("cropandweed", "cropandweed", crop_in, crop_in),
        ("cropandweed", "phenobench", crop_in, crop_in * 0.76),
        ("phenobench", "phenobench", pheno_in, pheno_in),
        ("phenobench", "cropandweed", pheno_in, pheno_in * 0.81),
    ]
    rows = []
    for train_dataset, test_dataset, in_domain, external in experiments:
        reliability = cross_dataset_summary(in_domain, external)
        rows.append(
            {
                "task": "classification",
                "model": "fake_reproducible_predictions",
                "train_dataset": train_dataset,
                "test_dataset": test_dataset,
                **reliability,
            }
        )
    return rows


def _reliability_rows(
    class_metrics: dict[str, float | str],
    cross_dataset: list[dict[str, float | str]],
) -> list[dict[str, float | str]]:
    rows = []
    for row in cross_dataset:
        rows.append(
            {
                **row,
                "primary_metric": "balanced_accuracy",
                "ece": class_metrics["ece"],
                "brier": class_metrics["brier"],
                "risk_coverage_auc": class_metrics["risk_coverage_auc"],
                "risk_at_80_coverage": class_metrics["risk_at_80_coverage"],
                "risk_at_90_coverage": class_metrics["risk_at_90_coverage"],
                "high_conf_wrong_rate": class_metrics["high_conf_wrong_rate"],
                "stressor": "clean",
                "latency_ms_batch1": np.nan,
                "fps_batch1": np.nan,
                "parameters_m": np.nan,
                "model_size_mb": np.nan,
            }
        )
    return rows


def _write_smoke_report(
    manifest_results: dict[str, dict],
    class_metrics: dict[str, float | str],
    segmentation_metrics: dict[str, float | str],
    cross_dataset: list[dict[str, float | str]],
    out_dir: Path,
) -> None:
    lines = ["# Smoke Test Report", ""]
    lines.append("This CPU-only smoke test validates the minimum benchmark pipeline.")
    lines.append("It does not train real models; prediction metrics are generated from fixed-seed fake predictions.")
    lines.append("")
    lines.append("## Manifest Status")
    lines.append("")
    for dataset, result in manifest_results.items():
        lines.append(f"- {dataset}: {len(result['records'])} records -> `{result['out_path'].relative_to(ROOT)}`")
        for warning in result["warnings"]:
            lines.append(f"  - WARNING: {warning}")
    lines.append("")
    lines.append("## Classification Reliability")
    lines.append("")
    for key in (
        "accuracy",
        "macro_f1",
        "balanced_accuracy",
        "ece",
        "brier",
        "risk_coverage_auc",
        "risk_at_80_coverage",
        "risk_at_90_coverage",
        "high_conf_wrong_rate",
    ):
        lines.append(f"- {key}: {float(class_metrics[key]):.4f}")
    lines.append("")
    lines.append("## Segmentation Smoke Metrics")
    lines.append("")
    for key in ("miou", "foreground_miou", "iou_background", "iou_crop", "iou_weed"):
        lines.append(f"- {key}: {float(segmentation_metrics[key]):.4f}")
    lines.append("")
    lines.append("## Cross-Dataset Template")
    lines.append("")
    lines.append(_df_to_markdown(pd.DataFrame(cross_dataset)))
    lines.append("")
    lines.append("## Generated Outputs")
    lines.append("")
    outputs = [
        "outputs/manifests/cropandweed_manifest.jsonl",
        "outputs/manifests/phenobench_manifest.jsonl",
        "outputs/metrics/dataset_summary.csv",
        "outputs/metrics/smoke_metrics.csv",
        "outputs/metrics/risk_coverage.csv",
        "outputs/metrics/cross_dataset_template.csv",
        "outputs/metrics/reliability_metrics_template.csv",
        "outputs/figures/reliability_diagram.pdf",
        "outputs/figures/risk_coverage_curve.pdf",
        "outputs/reports/manifest_report.md",
    ]
    for output in outputs:
        lines.append(f"- `{output}`")
    lines.append("")
    lines.append("## Next Step")
    lines.append("")
    lines.append("Replace fake predictions with one tiny real baseline after local CropAndWeed or PhenoBench paths are available.")
    (out_dir / "smoke_report.md").write_text("\n".join(lines), encoding="utf-8")


def _df_to_markdown(df: pd.DataFrame) -> str:
    columns = [str(col) for col in df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        values = [str(row[col]) for col in df.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
