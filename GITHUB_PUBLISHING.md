# GitHub and Zenodo publishing checklist

Target repository: https://github.com/Fytap/AgriReliabilityBench.git

The current public base release is `v1.0.0`. Because the revision artifacts on `main` post-date that tag, publish them as a new `v1.0.1` release rather than modifying the historical release.

Recommended commands after authenticating to GitHub:

```powershell
cd <local clone of AgriReliabilityBench>
git checkout main
git pull --ff-only origin main
git tag -a v1.0.1 -m "AgriReliabilityBench v1.0.1 revised reproducibility package"
git push origin v1.0.1
```

Before creating the archival release:

1. Confirm the author names, affiliations, and ORCID identifiers in `CITATION.cff` and `.zenodo.json`.
2. Verify that no raw datasets, large checkpoints, credentials, or private paths are committed.
3. Create a GitHub release from tag `v1.0.1`.
4. Create the linked Zenodo version and confirm its title, author order, version, and files.
5. Copy the generated version-specific DOI into `README.md`, `CITATION.cff`, `DATA_AND_CODE_AVAILABILITY.md`, and the manuscript Data and Code Availability statement.
6. Re-run the metadata validation after DOI insertion.
