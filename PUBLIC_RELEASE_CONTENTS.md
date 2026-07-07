# Public release contents

This package is intended to be uploaded as the manuscript's public reproducibility repository.

Included public materials:

- Code for dataset adapters, label harmonization, task derivation, metrics, stress corruptions, plotting, and aggregation.
- YAML configuration files for datasets, model families, experiment settings, and stressors.
- Published tables and supplementary CSVs used for manuscript review.
- Main and supplementary figures used to inspect calibration, stress robustness, external validation, inference reference, label harmonization, and failure audits.
- Major-revision derived outputs under `paper_artifacts/revision_outputs/`, including cross-dataset stress audit metrics, detection failure plate source files, CPU/ONNX timing outputs, and dataset/statistics audit files.
- Reproducibility reports, release audit files, environment files, and smoke-test workflow.

Excluded materials:

- Raw third-party datasets.
- Large model checkpoints.
- Full per-image prediction archives.
- Server-specific paths, credentials, and queue logs.

The package is therefore suitable for code/data availability in the sense of open code plus open derived results, while respecting the licenses of the original agricultural datasets.
