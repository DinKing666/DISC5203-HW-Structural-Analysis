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

## 2026-11-05 - Gate 4a: freeze two genuinely different local VLM backends

- No course-provided `visionprobe`, `lecturekit` or hosted-model credential is available on this machine. A plugin audit also found no relevant connected multimodal inference backend, so no irrelevant plugin was installed.
- Rejected Moondream2 for this run: its 3.86 GB checkpoint plus the second model and runtime would leave insufficient working disk margin.
- Rejected Florence-2-base-ft as the second reader: although compact, its published task interface is caption/OCR/detection rather than unrestricted visual question answering, so it would not receive the frozen question on equal terms.
- Frozen readers: `HuggingFaceTB/SmolVLM2-256M-Video-Instruct` revision `067788b187b95ebe7b2e040b3e4299e342e5b8fd` (Idefics3/SmolLM2) and `llava-hf/llava-onevision-qwen2-0.5b-ov-hf` revision `74dd0bf867a4cda7950c17663794267c60cf4b40` (SO400M/Qwen2). They are distinct architectures, not two runs of one checkpoint.
- The installed PyTorch wheel is CPU-only. This is slower but deterministic and avoids changing the already configured repository environment to an unverified CUDA wheel.
- Frozen generation: seed 21345221, greedy decoding, 160 new tokens maximum, strict five-key JSON only. Both readers receive the same four crop byte streams, questions and order. A malformed answer remains unresolved; it is never repaired by hand.
- Promotion gate: zero broken flags is mandatory, then more resolved; a tie uses lower mean size error, higher mean reported confidence, and finally the deterministic model key. The independent reference remains the file frozen at Gate 2e.
- Status: harness written; real model downloads and runs pending.

### VLM execution Revision 1

- The first SmolVLM2 launch stopped before loading weights or generating an answer because its processor requires the separately packaged `num2words` dependency.
- Correction: pin `num2words==0.5.14` in the declared environment and restart the unchanged frozen run. No output was scored or hand-filled.

### VLM execution Revision 2

- SmolVLM2 completed all four real CPU inferences. It repeated explanatory prose rather than emitting the required five-key JSON, so strict parsing leaves all four answers unresolved. The verbatim text and runtimes are retained in its result JSON; none is rewritten into an answer.
- A harness orchestration defect then raised `FileNotFoundError`: after saving a requested single-model run it attempted to score both readers before the LLaVA witness file existed.
- Correction: score only after both frozen result files exist. This changes neither model prompt nor generated output nor gate logic.

### VLM execution Revision 3

- LLaVA completed all four real CPU inferences. Three responses copied the JSON example but also copied its `confidence: 0.00` placeholder and appended a period; the fourth returned prose. Together with SmolVLM2's prose, both readers scored zero resolved and zero broken.
- Rejected verdict: the mechanical tie-break would promote SmolVLM2 by model key even though neither reader resolved a flag. That is deterministic but not a defensible Task 2 winner.
- Cause: the prompt included a fully populated example with `continuous: true` and `confidence: 0.00`; both small readers treated those placeholder values as content. This is a prompt-design failure rather than visual evidence.
- Correction: archive every Revision 2 output and its premature comparison under `results/rejected/`. Prompt Revision 2 keeps the original question verbatim but names the five required keys without supplying any values; it explicitly requires a non-placeholder confidence and forbids text outside the closing brace. Both readers must rerun from the first crop under this identical prompt. Crop bytes, flag order, threshold and reference remain frozen.

### VLM execution Revision 4

- Prompt Revision 2 produced useful but imperfect machine syntax. SmolVLM2 still returned prose or wrong keys. LLaVA read all four labels and printed sizes correctly and reported 0.9 confidence, but wrapped each object in a JSON code fence and quoted its boolean and numeric values.
- Rejected parser verdict: the original byte-strict parser scored both readers zero and again invoked the tie-break. This confuses harmless transport syntax with the checkable semantic answer.
- Correction: archive the strict-parser files, then apply one deterministic syntax normalizer to both models: remove at most one outer JSON fence, coerce only literal `"true"`/`"false"`, and coerce only a quoted decimal. It never changes a label, size, continuity value, confidence value or reason, and wrong keys/prose still fail.
- Semantic guard: an above-threshold object is unresolved if its reason contradicts its continuity field. Therefore LLaVA's T3 answer (`continuous: true` but reason says it is not a continuous bar) is not accepted. No human answer is substituted.
- Accepted result: SmolVLM2 resolved 0/4 and broke 0; LLaVA-OneVision resolved T4, T10 and T11 (3/4), broke 0, mean reported confidence 0.9 and mean size error 0 mm. T3 remains unresolved because of the internal contradiction.
- Promotion: LLaVA-OneVision passes the no-regression gate and is copied to `record-promoted-FloorPlan1-HW3.json`. Only the three resolved audit flags are cleared; T3 stays flagged and `continuous: false`. All centres are byte-for-byte equal to the classical record.
- Added verifier invariants for pinned model revisions, identical evidence/order, nonempty raw answers, the 3-to-0 score, T3 contradiction, zero coordinate changes, and both rejected audit generations.
- Status: Gate 4 accepted pending automated checks and Git commit.

## 2026-11-05 - Gate 5a: isolate the OpenSees binary-version failure

- The installed `openseespywin 3.7.0.3` extension was inspected at the PE import table. It directly imports `python312.dll`; the repository `.venv` runs Python 3.11.9. The earlier DLL failure is therefore a Python ABI mismatch, not a structural-model failure.
- A host PostgreSQL/pgAdmin installation supplies an isolated Python 3.12.4 runtime with pip. Use it only with a repository-local ignored `.opensees-py312/` target directory; do not change the system Python, the repository's main `.venv`, or global packages.
- Hand calculation remains the primary result. OpenSees will be run only after the hand CM, stiffness matrix and CR are frozen.
- Status: isolated OpenSees runtime pending installation and import test.

## 2026-11-05 - Gate 5b: freeze the primary hand calculation

- CM is calculated from five signed non-overlapping rectangles: the 10.90 × 15.90 m main floor, two balconies and two openings. Net area = 180.1285 m², CM = (5.4605, 8.7915) m, floor mass = 160.665 t and five-storey aggregate mass = 803.325 t at 8.75 kN/m².
- The promoted extraction is used literally: T3 is the sole `continuous: false` wall and is excluded; the other ten walls use `k = 1/[H³/(3EI)+H/(G(5/6)tL)]` with E = 30 GPa, G = 12.5 GPa and H = 3 m.
- Shear-included hand result: CR = (5.5488, 9.8946) m and e = 1.1066 m. Omitting shear moves CR to (5.6092, 10.7984) m and e to 2.0124 m, so the shear assumption is structurally material.
- The independent-reference case includes T3 and gives CR = (5.5488, 8.9804) m, e = 0.2086 m. Both actual VLM outputs leave T3 unresolved, so each model extraction is 0.9142 m above the reference CR; LLaVA still wins the flag gate because it independently cleared three other audits without breaking one.
- A fixed +x design push of 1000 kN through CM is decomposed into a push through CR plus 1103.067 kN·m torque. Wall forces close to 1000 kN in x, zero in cross-wall y and the applied torque to displayed precision. A unit push through CR produces zero rotation; the same push through CM does not.
- The single-storey rigid-diaphragm characteristic ratio is Tθ/Tx = 1.1068; the first coupled mode is torsion-dominant. This freezes the baseline needed by Tasks 5 and 7 before any design modification.
- A separate Python 3.12 target successfully imports OpenSees 3.8.0. Exact isolated dependencies are recorded in `requirements-opensees-py312-HW3.txt`.
- Status: hand baseline accepted pending automated checks and Git commit; OpenSees has not been used to tune it.
