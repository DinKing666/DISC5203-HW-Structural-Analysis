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

## 2026-11-05 - Gate 2a: environment audit

- The clean host Python does not provide Pillow or OpenSeesPy, so the first import audit stopped before image analysis.
- `visionprobe`, `lecturekit` and `ollama` are not currently discoverable on the command path.
- Decision: create a repository-local `.venv`, ignore it in Git, install the declared HW3 requirements and freeze the resolved versions. Do not modify the global Python installation.
- VLM rule: continue classical, forest and structural work independently; do not fabricate VLM outputs if two genuine model backends remain unavailable at the flag gate.
- Status: environment creation pending.

## 2026-11-05 - Gate 2b: first classical execution

- Revision 1 introduced the fixed threshold/morphology pipeline, two-pass RapidOCR label reading, connected-component score sheet and label-sized bar fitting.
- Failure: OpenCV `imread` could not open the valid FloorPlan1 image because the absolute repository path contains Chinese characters.
- Correction: read and write image bytes with NumPy plus `cv2.imdecode`/`cv2.imencode`; keep all paths and source bytes unchanged.
- Status: corrected implementation pending re-run.

## 2026-11-05 - Gate 2c: reject mislocated Revision 1 flags

- Revision 1 ran successfully and recovered all 18 named elements in both OCR passes, but it proposed four continuity flags for T2, T3, T4 and T11.
- Visual gate failed: the corresponding crops showed that the fixed-length search was still tied to the old HW1 full support segment instead of the short labelled wall. T4 was placed near C3 rather than at its lower 2 m hatch; T2 selected an unhatched bottom segment.
- Decision: reject all four Revision 1 flags and every downstream use of that record.
- Revision 2 uses the printed label position to bound the along-axis search while the inherited HW1 topology supplies only the nearby axis. The cross-axis coordinate becomes a support-weighted centre, and the column search radius is reduced from 0.30 m to 0.15 m to avoid neighbouring wall hatching.
- Determinism: stale flag crops are removed at the start of each run before the current manifest is written.
- Status: Revision 2 pending re-run and visual acceptance.

## 2026-11-05 - Gate 2d: distinguish text masking from structural support

- Revision 2 placed the wall axes correctly, including T4 at the lower-right 2 m hatch, but still marked five bars as low support.
- Diagnostic projection showed zero diagonal support through T2 even though the hatch is plainly visible. Cause: the cleaning step masked every OCR box from both passes, including low-confidence line fragments incorrectly recognised as text over structural bars.
- Correction: mask only schema-valid C/T size-label boxes. For vertical perimeter walls, intersect the label-anchored interval with the inherited topology neighbourhood so T10/T11 cannot extend above the main floor outline.
- Status: Revision 3 pending re-run and visual acceptance.

## 2026-11-05 - Gate 2e: accept Revision 3 and freeze the independent reference

- Revision 3 recovered all 18 element labels in each OCR pass and placed the 11 fixed-length wall bars on the visible hatching.
- Accepted flags are T3, T4, T10 and T11: the rule-derived diagonal support is incomplete at line/column intersections even though a manual reading sees a continuous printed-size wall.
- Frozen the independent reference before any VLM output. It uses a manual reading of hatching and printed size strings, with the accepted HW1 census only as a separate cross-check; it is not the kernel that generated the flags.
- Each crop is expanded to include both the flagged bar and its printed size label, and its SHA-256 is stored with the question manifest so both VLM runs receive identical evidence.
- Status: accepted pending deterministic re-run and automated checks.

## 2026-11-05 - Gate 2f: classical acceptance checks

- Added a stage-aware HW3 verifier and connected it to the cumulative repository's shared pre-commit hook after the accepted HW1 and HW2 checks.
- Required checks: frozen input hashes, exact L07 parameters, 7 columns plus 11 walls, unique label claims, two OCR passes, floor/load closure, exactly four accepted flags, crop hashes, and independent-reference agreement on the printed sizes.
- A second identical run must reproduce all tracked classical artifacts byte for byte before this gate is committed.
- Result: the second run reproduced all 21 tracked artifacts byte for byte. HW1, HW2 and the new stage-aware HW3 verifier all passed.
- Repository setup: enable the tracked `.githooks` directory through repository-local `core.hooksPath`; do not change global Git settings.
- Environment follow-up: the image/OCR stack is operational in `.venv`; the installed Windows OpenSeesPy binary currently fails to load a dependent DLL and remains a named issue for the structural-model gate.
- Status: Gate 2 accepted for Git commit.

## 2026-11-05 - Gate 3: kept-but-never-chosen random forest

- Freeze the accepted label-supervised classical record as the pseudo-label raster; the forest receives no independent ground truth.
- Features: grey, Sobel edge strength, ink density in 5/11/21/45 px windows and the 11 px 45-degree stroke response.
- Fixed model: 80 trees, maximum depth 14, minimum leaf size 3, balanced classes and random seed 21345221.
- Required evidence: serialised model, feature-importance CSV/plot, held-out metrics, full ROI prediction and rule-target IoU.
- Promotion prohibition is machine-readable (`selected: false`) and checked by `verify_hw3_outputs.py`.
- Status: training and acceptance pending.

### Forest Revision 1

- Held-out F1 was 0.9828 and the full-ROI rule-target IoU was 0.5757, demonstrating that the model mostly copied the pseudo-label appearance.
- Rejected feature interpretation: a single-pixel 45-degree boolean had importance 0.00014 because the response is sparse; 21 px and 45 px ink density dominated instead.
- Correction: retain the same 11 px diagonal morphology but represent its response as local 21 px stroke density, matching the intended neighbourhood feature. Hyperparameters, seed and pseudo-labels remain unchanged.
- Revision 2 result: held-out F1 = 0.9835, OOB score = 0.9830 and full-ROI rule-target IoU = 0.5848. The 21/45 px ink densities plus local 45-degree density carry 0.8402 of total feature importance.
- The 2.55 MB serialised forest is retained with its prediction and importance table. Its moderate full-plan IoU illustrates the catch: high held-out agreement with copied labels does not create an independent correction route.
- Determinism: a second training run reproduced the model, prediction masks, CSV, summary and figure byte for byte; the stage-aware verifier passed.
- Status: Revision 2 accepted for Git commit; selection remains prohibited.
