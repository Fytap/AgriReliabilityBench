# GitHub publishing checklist

Target repository: https://github.com/Fytap/AgriReliabilityBench.git

Recommended publish commands after installing Git and authenticating to GitHub:

```powershell
cd <local clone of AgriReliabilityBench_v1.0.0>
git init
git branch -M main
git remote add origin https://github.com/Fytap/AgriReliabilityBench.git
git add .
git commit -m "Release reproducibility package for IPA benchmark"
git push -u origin main
git tag v1.0.0
git push origin v1.0.0
```

Before creating the archival release:

1. Confirm author names and ORCID identifiers in `CITATION.cff` and `.zenodo.json`.
2. Confirm whether the repository should be public before journal submission.
3. Verify that no raw datasets, large checkpoints, credentials, or private paths are committed.
4. Create a GitHub release from tag `v1.0.0`.
5. If Zenodo is connected, copy the generated DOI into the manuscript Data and Code Availability statement.
