# Revision Dataset Audit

This audit is computed from the locally restored raw datasets and the generated benchmark manifests.
CWFID and CWD30 are external diagnostic test sets only and are not training or model-selection sources.

## Dataset Statistics

| dataset | split | images | masks | objects | crop objects | weed objects | weed-positive | weed-negative | crop pixel ratio | weed pixel ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cropandweed | unknown | 8034 | 8034 | 83130 | 23478 | 59652 | 5984 | 2050 | 0.056004 | 0.015818 |
| phenobench | test | 693 | 0 | 0 | 0 | 0 |  |  |  |  |
| phenobench | train | 1407 | 1407 | 22136 | 12475 | 9661 | 1360 | 47 | 0.118463 | 0.004971 |
| phenobench | val | 772 | 772 | 11384 | 6804 | 4580 | 738 | 34 | 0.095534 | 0.005283 |
| cwfid | test | 20 | 20 | 159 | 53 | 106 | 20 | 0 |  | 0.922777 |
| cwfid | train | 39 | 39 | 321 | 105 | 216 | 38 | 1 |  | 0.919023 |
| cwfid | train+test | 1 | 1 | 14 | 4 | 10 | 1 | 0 |  | 0.889582 |
| cwfid | all | 60 | 60 | 494 | 162 | 332 | 59 | 1 |  | 0.919784 |
| cwd30 | external | 196275 |  |  |  |  | 136064 | 60211 |  |  |

## CWD30 Class Collapse

CWD30 contributes 196275 images across 30 fine classes. The revision audit maps 10 fine classes to crop and 20 fine classes to weed.

## Manuscript Use

- Use the `all` CWFID row when describing external imbalance.
- Use PhenoBench train/val mask rows for pixel-ratio statements; official test masks are not present in the public release used here.
- Interpret CWD30 as classification-only external validation after crop/weed collapse.
