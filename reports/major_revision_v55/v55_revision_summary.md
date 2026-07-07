# v55 Major-Revision Progress Summary

## Completed in this pass

- Restored the reviewer-requested dataset-statistics evidence from raw datasets on the revision workspace.
- Added `supplementary/tableS_dataset_statistics.csv` with image, mask, object, class-balance, and pixel-ratio statistics where available.
- Updated `supplementary/tableS_label_mapping.csv` with the expanded crop/weed mapping, including all 30 CWD30 fine classes.
- Revised `main.tex` to report the main dataset counts, PhenoBench public-test-mask limitation, CWFID 59:1 imbalance, CWD30 classification-only scope, and the new supplementary table references.
- Redrew Figure 1 as a cleaner five-stage workflow schematic with color-coded stages, arrows, and a compact result-index strip.
- Rechecked Figure 1 after the table-order changes and removed fixed table numbers from the quick-index strip so the workflow figure cannot become stale if floats or earlier tables change.
- Reworked the main evidence, transfer, seed-stability, and external-validation tables with `tabularx` and normal column spacing to reduce column crowding and page-edge pressure.
- Added a manually checked representative YOLO11m CropAndWeed detection-failure plate with matched images, ground-truth labels, and prediction outputs.
- Added three-seed cross-dataset severity audits using formal 300-epoch checkpoints in both primary transfer directions: YOLO11m and RT-DETR-L for detection under blur, JPEG compression, and low-resolution stress, and SegFormer-B0 and Mask2Former-Swin-T for segmentation under blur, low-resolution, JPEG, brightness, and contrast stress.
- Added a representative CPU/ONNX detection inference-reference supplement: YOLO11m PyTorch CPU, YOLO11m ONNX Runtime CPU, and RT-DETR-L PyTorch CPU with 50 warm-up and 200 timed batch-1 runs.
- Updated the Results section to explain the diagnostic value of low external-validation scores, the agricultural implications of small-object and low-weed-coverage failures, and the supplementary detection-failure examples.
- Strengthened the Discussion and Limitations to clarify that the benchmark is a pre-deployment reliability profile, not a field trial, embedded deployment validation, chemical-dose study, yield-loss study, or closed-loop robot validation.
- Updated `reports/response_tracker_v55.tsv` and `reports/response_to_reviewers_draft_v55.md` so completed revisions are no longer described as pending or planned items.

## Current QA

- All `\includegraphics{}` and `\input{}` paths in `main.tex` resolve locally.
- No blocked cleanup terms or duplicated reference-heading strings were found in `main.tex`.
- The response draft no longer uses future-tense revision-plan wording for completed items.
- Figure reference labels in `main.tex` resolve locally.
- Table reference labels in `main.tex` and included table files resolve locally.
- Required supplementary CSV/PDF files for label mapping, box generation, training protocol, CI, multiseed audit, calibration, CWFID/CWD30 external validation, stress severity, detection/segmentation cross-dataset stress, CPU/ONNX inference, subgroup analysis, and detection-failure examples are present.
- Local LaTeX compilation was not run because `latexmk` and `pdflatex` are not installed on this Windows environment.

## Remaining evidence boundaries

- Local LaTeX compilation could not be run because `latexmk` and `pdflatex` are not installed in the current Windows environment; the package is prepared for Overleaf compilation and final float inspection.
- The revision now includes representative cross-dataset detection and segmentation severity audits, but it does not claim a full all-model environmental robustness simulator.
- The revision now includes representative CPU/ONNX detector timing, but it does not claim TensorRT, embedded-device, closed-loop robot, or field deployment validation.
- Additional external datasets are outside the present dataset scope; CWFID and CWD30 are retained as diagnostic external tests with explicit task limitations.
