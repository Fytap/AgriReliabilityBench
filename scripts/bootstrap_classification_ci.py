from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np


METRICS = ("accuracy", "balanced_accuracy", "macro_f1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-eval", default="outputs/metrics/formal_eval_long.csv")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--out-eval", default="outputs/metrics/classification_bootstrap_ci.csv")
    parser.add_argument("--out-retention", default="outputs/metrics/classification_retention_bootstrap_ci.csv")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = [row for row in read_csv(Path(args.formal_eval)) if row.get("task") == "classification"]

    eval_out = []
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for row in rows:
        pred_path = prediction_path(row["source_file"])
        if not pred_path.exists():
            continue
        labels, preds = load_predictions(pred_path)
        cache[row_key(row)] = (labels, preds)
        estimates = classification_metrics(labels, preds)
        intervals = bootstrap_metrics(labels, preds, args.n_bootstrap, rng)
        for metric in METRICS:
            eval_out.append(
                {
                    "run_id": pred_path.stem.removesuffix("_predictions"),
                    "model": row["model"],
                    "model_label": row.get("model_label", row["model"]),
                    "train_dataset": row["train_dataset"],
                    "test_dataset": row["test_dataset"],
                    "scope": row["scope"],
                    "metric": metric,
                    "estimate": estimates[metric],
                    "ci_low": intervals[metric][0],
                    "ci_high": intervals[metric][1],
                    "n_samples": len(labels),
                    "n_bootstrap": args.n_bootstrap,
                    "source_file": str(pred_path),
                }
            )

    retention_out = []
    indomain = {
        (row["model"], row["train_dataset"]): row
        for row in rows
        if row["scope"] == "in_domain" and row["train_dataset"] == row["test_dataset"]
    }
    for row in rows:
        if row["scope"] != "cross_dataset":
            continue
        base = indomain.get((row["model"], row["train_dataset"]))
        if not base:
            continue
        base_data = cache.get(row_key(base))
        ext_data = cache.get(row_key(row))
        if not base_data or not ext_data:
            continue
        retention_out.extend(
            bootstrap_retention(
                row,
                base,
                base_data,
                ext_data,
                args.n_bootstrap,
                rng,
            )
        )

    write_csv(Path(args.out_eval), eval_out)
    write_csv(Path(args.out_retention), retention_out)
    print(f"wrote {args.out_eval} rows={len(eval_out)}")
    print(f"wrote {args.out_retention} rows={len(retention_out)}")


def bootstrap_retention(
    row: dict[str, str],
    base: dict[str, str],
    base_data: tuple[np.ndarray, np.ndarray],
    ext_data: tuple[np.ndarray, np.ndarray],
    n_bootstrap: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    base_labels, base_preds = base_data
    ext_labels, ext_preds = ext_data
    base_values = bootstrap_metric_values(base_labels, base_preds, n_bootstrap, rng)
    ext_values = bootstrap_metric_values(ext_labels, ext_preds, n_bootstrap, rng)
    out = []
    for metric in METRICS:
        base_metric = base_values[metric]
        ext_metric = ext_values[metric]
        drops = base_metric - ext_metric
        ratios = np.divide(ext_metric, base_metric, out=np.full_like(ext_metric, np.nan), where=base_metric != 0)
        out.append(
            {
                "model": row["model"],
                "model_label": row.get("model_label", row["model"]),
                "train_dataset": row["train_dataset"],
                "external_test_dataset": row["test_dataset"],
                "metric": metric,
                "in_domain_estimate": classification_metrics(base_labels, base_preds)[metric],
                "external_estimate": classification_metrics(ext_labels, ext_preds)[metric],
                "drop_estimate": classification_metrics(base_labels, base_preds)[metric] - classification_metrics(ext_labels, ext_preds)[metric],
                "drop_ci_low": quantile(drops, 0.025),
                "drop_ci_high": quantile(drops, 0.975),
                "retention_estimate": classification_metrics(ext_labels, ext_preds)[metric] / classification_metrics(base_labels, base_preds)[metric],
                "retention_ci_low": quantile(ratios, 0.025),
                "retention_ci_high": quantile(ratios, 0.975),
                "n_in_domain": len(base_labels),
                "n_external": len(ext_labels),
                "n_bootstrap": n_bootstrap,
                "in_domain_source": prediction_path(base["source_file"]),
                "external_source": prediction_path(row["source_file"]),
            }
        )
    return out


def bootstrap_metrics(labels: np.ndarray, preds: np.ndarray, n_bootstrap: int, rng: np.random.Generator) -> dict[str, tuple[float, float]]:
    values = bootstrap_metric_values(labels, preds, n_bootstrap, rng)
    return {metric: (quantile(values[metric], 0.025), quantile(values[metric], 0.975)) for metric in METRICS}


def bootstrap_metric_values(labels: np.ndarray, preds: np.ndarray, n_bootstrap: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    n = len(labels)
    values = {metric: np.zeros(n_bootstrap, dtype=np.float64) for metric in METRICS}
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        metrics = classification_metrics(labels[idx], preds[idx])
        for metric in METRICS:
            values[metric][i] = metrics[metric]
    return values


def classification_metrics(labels: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    labels = labels.astype(int)
    preds = preds.astype(int)
    accuracy = float(np.mean(labels == preds))
    per_class_recalls = []
    per_class_f1 = []
    for cls in (0, 1):
        tp = float(np.sum((labels == cls) & (preds == cls)))
        fp = float(np.sum((labels != cls) & (preds == cls)))
        fn = float(np.sum((labels == cls) & (preds != cls)))
        denom_recall = tp + fn
        recall = tp / denom_recall if denom_recall else 0.0
        denom_f1 = (2.0 * tp) + fp + fn
        f1 = (2.0 * tp) / denom_f1 if denom_f1 else 0.0
        per_class_recalls.append(recall)
        per_class_f1.append(f1)
    return {
        "accuracy": accuracy,
        "balanced_accuracy": float(np.mean(per_class_recalls)),
        "macro_f1": float(np.mean(per_class_f1)),
    }


def load_predictions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    labels = []
    preds = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            labels.append(int(float(row["label"])))
            preds.append(int(float(row["pred"])))
    return np.asarray(labels, dtype=np.int64), np.asarray(preds, dtype=np.int64)


def prediction_path(source_file: str) -> Path:
    source = Path(source_file)
    return source.with_name(source.stem + "_predictions.csv")


def row_key(row: dict[str, str]) -> str:
    return "|".join([row["model"], row["train_dataset"], row["test_dataset"], row["scope"]])


def quantile(values: np.ndarray, q: float) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    return float(np.quantile(values, q))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
