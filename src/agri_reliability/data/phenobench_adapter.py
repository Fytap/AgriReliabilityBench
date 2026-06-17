from __future__ import annotations

from agri_reliability.data.dataset_adapter import AdapterResult, build_manifest_from_config


def build_phenobench_manifest(
    config: dict,
    *,
    max_records: int | None = None,
    derive_from_masks: bool = True,
) -> AdapterResult:
    return build_manifest_from_config(
        config,
        task="classification_detection_segmentation",
        max_records=max_records,
        derive_from_masks=derive_from_masks,
    )
