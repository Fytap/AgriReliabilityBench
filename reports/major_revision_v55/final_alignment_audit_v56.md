# Final alignment audit after pre-submission consistency review

Date: 2026-07-07

## Manuscript fixes completed

- Added explicit Methods wording that training data volume was not artificially equalized between CropAndWeed and PhenoBench.
- Clarified that PhenoBench-target rows use the mask-available validation split because public test masks are not distributed in the release used here.
- Clarified that detection rows in the three-seed audit use mAP50, while the compact official transfer table reports mAP50--95.
- Clarified that clean scores in severity-sweep audits are subset-based audit baselines, not replacements for official full-evaluation scores.
- Updated Figure 14 caption to point readers to Supplementary Figure S-detection-failure-plate for detection qualitative examples.
- Reduced the density of the related-work comparison table and added an explicit TORA discussion sentence.
- Confirmed no unresolved double-period Method headings remained in the manuscript scan.
- Standardized segmentation model names in main tables and text to avoid ambiguous bare Swin-T labels.

## Supplementary package checks

All referenced high-priority supplementary files were present after the final pass:

- Supplementary Table S-dataset-statistics
- Supplementary Table S-label
- Supplementary Table S-box
- Supplementary Table S-training-protocol
- Supplementary Table/Figure S-detection-failure-plate
- Supplementary Table/Figure S-cross-dataset-detection-stress-severity-full
- Supplementary Table/Figure S-cross-dataset-segmentation-stress
- Supplementary Table S-CPU-ONNX-inference-reference
- Supplementary Table S-temperature-scaling
- Supplementary Table S-subgroup-analysis
- Supplementary Tables S-CWFID and S-CWD30

## Figure QA

Key figure PDFs were rendered to local QA previews. The pipeline, severity-sweep, Pareto, classification-failure, and segmentation/stress-failure figures were visually inspected. Pareto figures retain centered numbered blue markers without leader lines; marker size was slightly reduced to reduce crowding while preserving the original style and data coordinates.

## Response-letter alignment

The submit-ready response letter was updated with more concrete manuscript locations, including Section 3.1 for dataset/split/training-volume clarification, Section 4.4 for cross-dataset stress, Section 4.7 for inference reference, Section 4.8 for failure/subgroup evidence, Table 7 for external-validation interpretation, and named supplementary tables/figures for added evidence.

## Remaining compile note

No local LaTeX compiler is installed in the current Windows environment. The package should be compiled and visually checked in Overleaf before upload.
