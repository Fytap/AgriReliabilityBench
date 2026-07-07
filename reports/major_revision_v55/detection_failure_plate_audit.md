# Detection Failure Plate Audit

The reviewer requested representative detection failure visualizations. This item is completed with matched images, ground-truth labels, and prediction outputs from a representative YOLO11m CropAndWeed audit.

## Completed artifacts

- `supplementary/figureS_detection_failure_plate.pdf`
- `supplementary/tableS_detection_failure_plate_index.csv`

## Verification basis

The plate was generated from restored raw CropAndWeed images, matched label files, and YOLO11m prediction outputs produced in the revision workspace. The examples are used as illustrative qualitative evidence and are not treated as exhaustive quantitative results.

## Failure modes shown

- Missed weed objects.
- Low-overlap or shifted localization.
- False crop/weed objects.

## Claim discipline

The manuscript keeps detection conclusions anchored in aggregate external-validation, stress, and CI tables. The qualitative plate is described as illustrative rather than exhaustive.
