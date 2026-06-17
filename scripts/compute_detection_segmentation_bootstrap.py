"""Compute detection/segmentation bootstrap CI once matched records are available.

Inputs required:
- segmentation: per-image prediction masks and unified ground-truth masks;
- detection: COCO ground-truth JSON and COCO prediction JSON with matching image IDs.
Outputs follow tableS_segmentation_bootstrap_ci.csv and tableS_detection_bootstrap_ci.csv schemas.
"""

raise SystemExit("Matched records are required before CI computation.")
