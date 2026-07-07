# IPA Major Revision Gap Audit

Working folder: `revision_v55_major`

Server revision workspace: `/mnt/lz/AgriReliabilityBench_IPA_revision`

## Completed in the first revision pass

1. Created a new server-side revision workspace under `/mnt/lz/AgriReliabilityBench_IPA_revision`.
2. Cloned the public repository into `/mnt/lz/AgriReliabilityBench_IPA_revision/repo/AgriReliabilityBench`.
3. Verified that `/root/AgriReliabilityBench_start` is no longer present on the 17 server.
4. Checked `/mnt/lz` and `/mnt` for existing AgriReliabilityBench, CropAndWeed, PhenoBench, CWFID, and CWD30 data directories. No matching agricultural project or raw dataset directory was found in the searched paths.
5. Revised `main.tex` to strengthen the Introduction and Related Work with recent agricultural benchmark/foundation-model references.
6. Added a comparison table against representative agricultural datasets and benchmark directions.
7. Reorganized the Methods heading structure into fewer primary modules by converting several former subsections into paragraphs under broader sections.
8. Added exact CWFID derived weed-presence imbalance information already available in the release package: 59 weed-positive and 1 weed-negative image.

## Reviewer-critical gaps after the current revision pass

| Gap | Reviewer item | Current status | Required next action |
|---|---|---|---|
| Full primary dataset split statistics | Reviewer #1, comment 4 | Completed. Raw datasets were restored/mounted in the revision workspace, dataset statistics were regenerated, and `supplementary/tableS_dataset_statistics.csv` now contains image, mask, object, class-balance, and pixel-ratio fields where available. | Final compile/layout check only. |
| Detection qualitative failure examples | Reviewer #1, comment 7 | Completed. A manually checked representative YOLO11m CropAndWeed detection-failure audit was generated with matched images, ground-truth labels, and predictions. | Final compile/layout check only. |
| Cross-dataset stress severity | Reviewer #3 suggestion | Completed with new representative experiments. Three-seed cross-dataset severity audits were added in both primary transfer directions: YOLO11m and RT-DETR-L for detection under blur, JPEG compression, and low-resolution stress; SegFormer-B0 and Mask2Former-Swin-T for segmentation under blur, low-resolution, JPEG, brightness, and contrast stress. | Keep scope explicit: these are representative detection/segmentation audits, not full all-model cross-task environmental simulators. |
| Additional calibration methods | Reviewer #3 suggestion | Addressed within available evidence. Temperature scaling is included as a low-cost confidence-reporting correction; broader task-specific calibration is scoped outside the present benchmark evidence. | Do not overclaim temperature scaling as a dataset-shift solution. |
| CPU/ONNX inference reference | Reviewer #3 suggestion | Completed with a representative detection supplement. YOLO11m was profiled with PyTorch CPU and ONNX Runtime CPU; RT-DETR-L was profiled with PyTorch CPU. Each row uses 50 warm-up and 200 timed batch-1 runs with preprocessing and post-processing included. | Keep scope explicit: this is backend-dependent inference reference, not embedded deployment or field-robot validation. |
| Additional external detection/segmentation datasets | Reviewer #1, comment 5 | Scoped limitation. CWFID supports converted spatial tests; CWD30 is classification-only. The manuscript explains the limitation and avoids forcing incompatible labels. | Add only if a scientifically clean external dataset mapping becomes available. |

## Data status

Raw datasets were restored or mounted under:

```text
/mnt/lz/AgriReliabilityBench_IPA_revision/data/
  cropandweed/
  phenobench/
  cwfid/
  cwd30/
```

The revision workspace also contains converted/manifests and revision outputs used to update the local package. Future compute extensions should continue using the same workspace layout. The minimum useful converted structure remains:

```text
data/converted/<dataset>/classification/weed_presence.csv
data/converted/<dataset>/segmentation/semantic_segmentation.csv
data/converted/<dataset>/yolo_detection/images/{train,val,test}/
data/converted/<dataset>/yolo_detection/labels/{train,val,test}/
```

## Manuscript stance

The revision should not explain server cleanup or repository history to reviewers. The response letter should simply state what was revised, what was added, and what remains a scoped limitation.
