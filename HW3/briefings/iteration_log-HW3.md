# HW3 iteration and gate record

## 2026-11-05 - Gate 0: recover the correct cumulative repository

- Confirmed the course source of truth is `https://github.com/DinKing666/DISC5203-HW-Structural-Analysis`, branch `main`, starting commit `fe010445`.
- Confirmed the cumulative convention is one Git history with separately reviewable `HW1/`, `HW2/` and future `HWn/` packages.
- Compared the obsolete local HW2 working tree with cumulative `HW2/`. The substantive 45/60/30 tributary correction is already present; remaining differences are cumulative paths, filenames, wording and a more robust PDF check.
- Re-ran the accepted baselines. HW1 passed with 18 elements, exact area closure and exact global load conservation. HW2 passed with 24 beams, four equilibrium checks and an 11-page PDF.
- Decision: do not modify or commit the obsolete HW2 repository. Extend only the clean cumulative repository.
- Status: accepted.

## 2026-11-05 - Gate 1: freeze HW3 assignment and course assets

- Student ID 21345221 selects `FloorPlan1.png`.
- Copied the supplied Markdown assignment, PDF assignment and asset archive into `HW3/assignment/` with the cumulative naming convention.
- Expanded the untouched course images into `HW3/assets/` and recorded their byte counts and SHA-256 digests in `config/input_manifest-HW3.json`.
- Added the plain-language pipeline, project rules and reusable procedures before analysis code is written.
- No image extraction, OCR, VLM answer or structural result exists at this gate.
- Check: all eight frozen files match their recorded byte counts and SHA-256 digests and no file exceeds 6 MB. The supplied immutable assignment Markdown contains four original trailing-space lines; its bytes are preserved, while authored files pass `git diff --check`.
- Git iteration: the first Gate 1 commit attempt was correctly rejected because this clean clone had no author identity. Configure the repository-local identity from the author's existing published commits; do not change global Git configuration.
- Status: accepted for the Gate 1 Git commit.
