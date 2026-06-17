# Reproducibility guide

This repository supports four levels of reproducibility.

## Level 1: smoke test

Runs without raw datasets or GPU access and checks the core metric/reporting path.

```bash
python scripts/run_smoke_test.py
```

## Level 2: manifest reproduction

Requires the official datasets under `data/raw/`.

```bash
python scripts/make_manifest.py --config configs/datasets/cropandweed.yaml
python scripts/make_manifest.py --config configs/datasets/phenobench.yaml
```

## Level 3: model training and evaluation

Requires datasets, model dependencies, and GPU resources. Representative entry points include:

```bash
python scripts/train_classification_timm.py --help
python scripts/train_segmentation_smp.py --help
python scripts/eval_classification.py --help
python scripts/eval_rtdetr_ultralytics.py --help
python scripts/eval_segmentation_smp.py --help
```

The paper used established model families rather than new architectures. Training protocols and key hyperparameters are summarized in `paper_artifacts/supplementary/tableS_training_protocol.csv`.

## Level 4: paper-result audit

The public package includes derived metric tables, supplementary CSVs, figure assets, and release audits. Exact per-image prediction reconstruction may require local access to raw datasets and non-redistributed checkpoints or prediction archives.

Important interpretation notes:

- Official aggregate metrics are the ranking scores reported by task-specific evaluators.
- Bootstrap means can differ from official metrics because they are computed from resampled records.
- External validation is a diagnostic stress test, not expected performance-preserving transfer.
- Batch-1 inference numbers are an inference reference, not field deployment validation.
- Temperature scaling is a confidence calibration correction and does not solve dataset shift.
