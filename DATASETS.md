# Dataset access and local layout

Raw datasets are not redistributed in this repository. Users must obtain each dataset from its official provider and comply with its license and access terms.

Expected local layout:

```text
data/raw/
  cropandweed/
  phenobench/PhenoBench/
  cwfid/
  cwd30/
```

The authoritative relative roots, image directories, annotation directories, and label mappings are defined in `configs/datasets/`.

## Dataset roles in the manuscript

| Dataset | Role | Tasks |
|---|---|---|
| CropAndWeed | Primary training and evaluation | classification, detection, segmentation |
| PhenoBench | Primary training and evaluation | classification, detection, segmentation |
| CWFID | External diagnostic testing only | compatible classification, detection, segmentation diagnostics |
| CWD30 | External diagnostic testing only | classification after crop/weed category collapse |

## Data redistribution policy

Do not commit raw images, annotations, masks, checkpoints, or complete prediction archives to this repository. The repository includes empty `data/raw/` and `data/processed/` roots only so users can reproduce the expected local structure.

Derived aggregate metric tables and manuscript figure assets are included under `paper_artifacts/` for review and reproducibility auditing.
