from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


TASK_COLORS = {
    "classification": "#2b6cb0",
    "segmentation": "#2f855a",
    "detection": "#c05621",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-dir", default="outputs/metrics")
    parser.add_argument("--out-dir", default="outputs/figures")
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reliability = pd.read_csv(metrics_dir / "paper_ready_reliability_deployment_summary.csv")
    stress_path = metrics_dir / "classification_stress_summary.csv"
    stress = pd.read_csv(stress_path) if stress_path.exists() else None
    seg_stress_path = metrics_dir / "segmentation_stress_summary.csv"
    seg_stress = pd.read_csv(seg_stress_path) if seg_stress_path.exists() else None
    det_stress_path = metrics_dir / "detection_stress_summary.csv"
    det_stress = pd.read_csv(det_stress_path) if det_stress_path.exists() else None

    plot_retention(reliability, out_dir / "cross_dataset_retention.pdf")
    plot_drop(reliability, out_dir / "cross_dataset_absolute_drop.pdf")
    plot_deployment_tradeoff(reliability, out_dir / "deployment_latency_vs_external_score.pdf")
    if stress is not None:
        plot_classification_stress(stress, out_dir / "classification_stress_balanced_accuracy_drop.pdf")
    if seg_stress is not None:
        plot_stress_heatmap(
            seg_stress,
            "miou_absolute_drop",
            "Segmentation mIoU drop under stress",
            out_dir / "segmentation_stress_miou_drop.pdf",
        )
    if det_stress is not None:
        plot_stress_heatmap(
            det_stress,
            "mAP50-95_absolute_drop",
            "Detection mAP50-95 drop under stress",
            out_dir / "detection_stress_map50_95_drop.pdf",
        )

    print(f"wrote figures to {out_dir}", flush=True)


def plot_retention(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(8.0, 8.5), sharex=False)
    for ax, task in zip(axes, ["classification", "segmentation", "detection"]):
        sub = df[df["task"] == task].copy()
        sub["label"] = sub["model_label"] + "\n" + sub["train_dataset"] + " to " + sub["external_test_dataset"]
        ax.bar(range(len(sub)), sub["retention"], color=TASK_COLORS[task], alpha=0.85)
        ax.axhline(1.0, color="#333333", linewidth=0.8, linestyle="--")
        ax.set_title(task.capitalize(), fontsize=10)
        ax.set_ylabel("Retention")
        ax.set_ylim(0, max(1.05, float(sub["retention"].max()) * 1.15))
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(sub["label"], rotation=35, ha="right", fontsize=7)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_drop(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), sharey=False)
    for ax, task in zip(axes, ["classification", "segmentation", "detection"]):
        sub = df[df["task"] == task].copy()
        sub["label"] = sub["model_label"] + "\n" + sub["train_dataset"] + " to " + sub["external_test_dataset"]
        ax.bar(range(len(sub)), sub["absolute_drop"], color=TASK_COLORS[task], alpha=0.85)
        ax.set_title(task.capitalize(), fontsize=10)
        ax.set_ylabel("Absolute drop")
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(sub["label"], rotation=45, ha="right", fontsize=7)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_deployment_tradeoff(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for task, sub in df.groupby("task"):
        ax.scatter(
            sub["deployment_latency_ms"],
            sub["external_score"],
            s=45 + 18 * sub["deployment_params_millions"].clip(upper=90) / 10,
            label=task,
            color=TASK_COLORS.get(task, "#444444"),
            alpha=0.8,
            edgecolor="white",
            linewidth=0.6,
        )
        for _, row in sub.iterrows():
            label = f"{row['model']}\n{row['train_dataset'][:2]}->{row['external_test_dataset'][:2]}"
            ax.annotate(label, (row["deployment_latency_ms"], row["external_score"]), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Batch=1 latency (ms)")
    ax.set_ylabel("External score")
    ax.set_title("External reliability vs deployment latency", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_classification_stress(df: pd.DataFrame, out: Path) -> None:
    sub = df[df["corruption"] != "clean"].copy()
    pivot = sub.pivot_table(
        index="run_id",
        columns="corruption",
        values="balanced_accuracy_absolute_drop",
        aggfunc="mean",
    )
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    image = ax.imshow(pivot.values, aspect="auto", cmap="RdYlBu_r")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_title("Classification balanced-accuracy drop under stress", fontsize=10)
    cbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Drop from clean")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def plot_stress_heatmap(df: pd.DataFrame, metric: str, title: str, out: Path) -> None:
    sub = df[(df["corruption"] != "clean") & df[metric].notna()].copy()
    if sub.empty:
        return
    pivot = sub.pivot_table(index="run_id", columns="corruption", values=metric, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    image = ax.imshow(pivot.values, aspect="auto", cmap="RdYlBu_r")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_title(title, fontsize=10)
    cbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Drop from clean")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


if __name__ == "__main__":
    main()
