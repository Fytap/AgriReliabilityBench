# AgriReliabilityBench

Reproducibility package for the manuscript:

**Reproducible reliability benchmarking for crop-weed visual perception under dataset shift**

This repository is prepared for paper review and public reproducibility. It contains the benchmark code, configuration files, derived result tables, figure assets, supplementary materials, and audit reports needed to inspect the reported experiments. It does **not** redistribute raw datasets or large trained checkpoints.

Repository: https://github.com/Fytap/AgriReliabilityBench

Release version: `1.0.1`

The version-specific Zenodo DOI is issued when the `v1.0.1` GitHub release is archived. It is then recorded in this README, `CITATION.cff`, the Zenodo record, and the manuscript Data and Code Availability statement.

## What this repository supports

AgriReliabilityBench evaluates whether agricultural vision models remain reliable when moving between datasets, task formulations, label levels, image conditions, and batch-1 inference constraints. The paper is a benchmark and decision-support study, not a new-model paper.

Main benchmark components:

- **Primary datasets:** CropAndWeed and PhenoBench.
- **External diagnostic datasets:** CWFID and CWD30.
- **Tasks:** weed-presence classification, crop/weed detection, and semantic segmentation.
- **Reliability evidence:** in-domain performance, cross-dataset transfer and retention, calibration, selective prediction, stress robustness, external validation, bootstrap confidence intervals, three-seed audit, and batch-1 inference reference.

## Repository map

```text
configs/                 Dataset, model, experiment, and stress configuration files
src/agri_reliability/    Dataset adapters, label mapping, metrics, stressors, and reporting utilities
scripts/                 Manifest, training, evaluation, calibration, bootstrap, stress, and figure scripts
data/                    Empty local data roots; raw datasets are intentionally excluded
paper_artifacts/         Main tables, figures, and supplementary result tables used by the manuscript
paper_artifacts/revision_outputs/
                         Derived major-revision audit outputs, small CSV/JSON metrics, and qualitative plates
reports/                 Release audit, file manifest, and reproducibility reports
.github/workflows/       Lightweight smoke-test workflow
```

## What is included

- Source code for manifest generation, label harmonization, derived task construction, metrics, stressors, and plotting.
- Configuration files for the paper datasets and benchmark protocols.
- Published aggregate tables, supplementary CSVs, qualitative figure assets, and audit reports.
- Major-revision derived experiment outputs, including cross-dataset stress audits, detection failure plates, CPU/ONNX timing summaries, and dataset/statistics audits.
- Environment files and command templates for reproducing the pipeline after obtaining the licensed datasets.
- A smoke test that runs without GPU and without raw datasets.

## What is not included

- Raw CropAndWeed, PhenoBench, CWFID, or CWD30 images and annotations.
- Large model checkpoints, full prediction archives, and server-specific training queues.
- Private paths, credentials, SSH keys, or dataset copies.

The raw datasets must be obtained from their original providers and placed locally according to `DATASETS.md`.

## Quick start for reviewers

```bash
conda env create -f environment.yml
conda activate agri-rel
pip install -e .
python scripts/run_smoke_test.py
```

The smoke test validates the repository structure, metric code path, and report generation without requiring GPU access or raw datasets.

## Reproducing the full benchmark

1. Obtain the official datasets listed in `DATASETS.md`.
2. Place them under the local `data/raw/` layout described there.
3. Generate manifests:

```bash
python scripts/make_manifest.py --config configs/datasets/cropandweed.yaml
python scripts/make_manifest.py --config configs/datasets/phenobench.yaml
```

4. Run task-specific training/evaluation scripts or inspect the published derived result tables in `paper_artifacts/`.

Full training requires substantial GPU resources and is described in `REPRODUCIBILITY.md`.

## Scientific interpretation

External validation should be interpreted as a diagnostic stress test rather than an expected performance-preserving transfer setting. Batch-1 timing is an inference reference, not field deployment validation. Temperature scaling is a low-cost confidence-reporting correction, not a dataset-shift solution.

## Citation

If you use this repository, cite the archived release metadata in `CITATION.cff` and the version-specific Zenodo DOI listed in the release notes. The author order, affiliations, and ORCID identifiers are aligned with the accompanying manuscript.

## License

The software components in this repository are released under the MIT License. Third-party datasets and model dependencies remain governed by their original licenses.
