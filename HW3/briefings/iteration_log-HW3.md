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

## 2026-11-05 - Gate 5c: first isolated OpenSees launch

- The Python 3.12 runner reached the structural script but stopped before model creation because pgAdmin's isolated `python312._pth` omits the invoked script directory from `sys.path`; the sibling `structural_params.py` could not be imported.
- Correction: explicitly prepend the resolved HW3 script directory after the ignored package target. No model assumption or frozen hand number changes.
- Status: corrected launch pending re-run.

## 2026-11-06 - Gate 5d: accept the pertinent OpenSees model

- Built one 3D `ElasticTimoshenkoBeam` from base to floor for each of the ten promoted continuous walls. The local strong axis follows the plan wall length; strong-axis shear area is `(5/6)tL`. Weak-axis action and St Venant wall torsion are set to 1e-9 of the strong idealisation so the model matches the hand premise that each wall resists only in its plane.
- Declared `rigidDiaphragm(3, master, *wall_top_nodes)` at CM with free `ux`, `uy`, `rz`, and used the `Transformation` constraint handler.
- Three unit loads at CM generated F directly and K by inversion. OpenSees 3.8.0 gives CR = (5.5488445701, 9.8945888601) m: 9.33e-10 m from the frozen hand CR. Maximum 3×3 K relative difference is 9.24e-9.
- The unit +x push applied through CR produces zero rotation to output precision. The 1000 kN CM case is reproduced by the push-through-CR case plus the eccentric torque case wall by wall; maximum additivity residual is zero.
- OpenSees wall forces agree with every hand row within 1e-5 kN. Their x sum is 999.9999989 kN and cross-wall y sum is 2.17e-8 kN.
- Determinism: a complete second build and all seven load cases reproduced the 33,967-byte JSON exactly, SHA-256 `51d9c2ed4dcb8e4d334cc8abafaec06c6c52c919f84d302e8b23533b12d4fa5a`.
- Status: baseline model accepted pending automated checks and Git commit.

## 2026-11-06 - Gate 6: make CM and CR coincide without moving the grid

- Chosen modification keeps the slab mass regions, columns, openings and all wall centrelines fixed. T3 is rebuilt as a continuous 309 × 2000 mm wall on its extracted line; T7 is thickened from 250 to 365 mm toward the shaft side.
- The two thicknesses come from independent stiffness-weighted coordinates: horizontal-wall stiffness controls `y_CR`, vertical-wall stiffness controls `x_CR`. Millimetre rounding is applied only after solving the exact required stiffness.
- Modified hand result: CR = (5.4606070, 8.7917400) m versus CM = (5.4604506, 8.7915223) m. Eccentricity falls from 1.106603 m to 0.000268 m (99.9758 % reduction); the residual is 0.16 mm in x and 0.22 mm in y.
- For the fixed 1000 kN +x design push, eccentric torque falls from 1103.067 to 0.218 kN·m and peak absolute wall force falls from 604.080 to 582.194 kN. Baseline T2 was most torsion-amplified at 1.3260 times its no-turn push share.
- The characteristic Tθ/Tx ratio changes from 1.1068 to 1.0266. The torsion mode remains first but becomes 99.9999 % pure after CM/CR decoupling; coincidence removes modal mixing, not necessarily the ordering set by mass and stiffness.
- Modified OpenSees gives CR = (5.4606069847, 8.7917399865) m, 1.55e-9 m from the modified hand value. Unit push through CR again has zero rotation; design wall-force sums and push-plus-turn additivity pass.
- Architectural note: no wall is relocated into either opening and no column/grid line is disturbed. The exercise holds mass fixed, so the small wall self-weight change is excluded consistently from both before and after CM.
- Determinism: the modified hand JSON/CSV/PNG and both baseline/modified OpenSees JSON files reproduced byte for byte on complete re-runs. Modified hand SHA-256 is `ff04e3888041b89add20e014675654cc49f00bc7126b927995f5fc65efaa4719`; modified OpenSees SHA-256 is `3bafffe4258e6ec10b98fecb85af26a814ef3ce67adf562b4a9caa731d0403b7`.
- Status: modification accepted pending automated checks and Git commit.

## 2026-11-06 - Gate 7a: freeze the two untuned-plan sources and references

- `NewPlan.png` is the course-supplied unseen plan. Its SHA-256 is frozen before reuse of the FloorPlan1 reader. The printed 46 m by 18 m grid, global `C1:C30 = 800 mm x 800 mm` note and eight wall callouts form a manual independent reference; the OCR/morphology does not create this reference.
- First found-plan candidate: the City of Pleasanton public single-family sample, PDF page 9 / sheet S-4. It is a wood shear-wall drawing whose schedule identifies nailing systems rather than one concrete wall size per element. Reject it because forcing the assignment's concrete stiffness model onto that sheet would change both the label premise and mechanics.
- Accepted found plan: MLStructFP public test sample Floor 966, retained with its public test annotation and MIT license. It is a multi-unit structural floor plan at 366.506 px/m, with an irregular slab, 20 wall groups represented by 28 rectangles, wall thicknesses of 200/250/300 mm, and one 8.5-degree wall.
- The found plan deliberately differs from FloorPlan1: transparent coloured-line CAD rather than black hatching, no printed C/T size strings and a non-orthogonal segment. This is a meaningful generalisation test rather than another plan tuned to the fixed reader.
- The dataset JSON reference is frozen before any VLM output. It is an independent public annotation, explicitly treated as a defensible reference rather than absolute ground truth.
- No VLM crop or answer has been produced at this gate. Threshold 225, 11-px diagonal kernel, 17/21-px openings, 4:1 shape rule, 0.85 OCR threshold and the existing confidence/gate rules remain locked.
- Status: sources and independent references pending hash verification and Git commit.

## 2026-11-06 - Gate 7b: NewPlan classical pass, Revision 1

- The first untuned execution stopped before writing an accepted record. T1 is split by OCR into an isolated `1` and a nearby `3500x400` box; the same size box appears once in each OCR pass, so a rule requiring exactly one token/size pair saw two duplicate candidates.
- Correction: retain the fixed 0.85 OCR threshold and pair geometry, but deterministically deduplicate the two-pass evidence by highest minimum pair confidence, then nearest x/y distance and pass name. This changes no pixel coordinate, printed dimension, morphology parameter or independent reference.
- Status: Revision 2 pending re-run and visual audit.

### NewPlan classical Revision 2

- Revision 2 stopped at the label-completeness assertion: removing all spaces before the wall regex changed `T2 3500x400` to `T23500x400`, making the wall number and length ambiguous. This normalization was only needed for noisy column strings such as `IIC11`.
- Correction: apply the wall label/size regex to the original OCR text, while retaining compact matching only for column IDs. OCR bytes, acceptance threshold, evidence pairing and all geometric rules remain unchanged.
- Status: Revision 3 pending re-run and visual audit.

### NewPlan classical Revision 3

- Revision 3 recovered all 38 expected labels, then stopped before element output because the reused FloorPlan1 helper expects the full label dictionary and the new caller supplied one label record. The helper correctly rejected the incompatible call with `KeyError` rather than silently using a wrong size.
- Correction: pass the already frozen full label dictionary to both reused element helpers. No reading, parameter or result is altered.
- Status: Revision 4 pending re-run and visual audit.

### NewPlan classical Revision 4 visual rejection

- Revision 4 produced 38 named elements and one label flag, but the overlay failed the geometry gate. The initial manual topology read T3/T7 from grid line 3 downward; the drawn black bars actually rise 3.5 m from grid line 4. Correct the manual reference and topology before any VLM is run.
- T2 and T5 were displaced by about one metre because their adjacent callouts are not centred on the bars. The inherited horizontal helper lacked the topology fallback already present for vertical bars. Make the search symmetric: intersect label and printed-grid neighbourhoods when a full labelled length fits; otherwise search the printed-grid neighbourhood. Apply the same generic rule to both plans and require the FloorPlan1 artifacts to remain byte-identical.
- T1's split OCR evidence was represented by a union box for cropping, but using that union as one text mask erased the wall pixels between the true `1` and size boxes. Keep the union only for crop extent and mask the two actual OCR boxes independently.
- These corrections change no locked numerical parameter and occur before either VLM sees NewPlan.
- Status: Revision 5 pending re-run, baseline non-regression check and visual acceptance.

### NewPlan classical Revision 5 baseline regression

- Applying the symmetric topology fallback globally changed FloorPlan1 from four to five flags and rewrote its frozen crop/record evidence. The mandatory no-regression check rejected this change before NewPlan was accepted.
- Correction: restore the original FloorPlan1 behaviour as the helper default. Expose the same topology fallback only as an explicit call for NewPlan, whose callouts are demonstrably adjacent to rather than centred on T2/T5. FloorPlan1 must return to a byte-clean state after its complete rerun.
- Status: Revision 6 pending baseline restoration, NewPlan rerun and visual acceptance.

### NewPlan classical Revision 6 acceptance

- The complete FloorPlan1 classical rerun returned every tracked baseline artifact to its committed byte state, so the optional NewPlan topology fallback does not regress the main plan.
- Revision 6 recovers all 38 named elements: 30 columns inherit the printed global 800 x 800 mm size note and eight walls use their own printed callouts. The two OCR passes each contribute 35 unique labels; their union plus the auditable split T1 reconstruction closes the census.
- The fixed rules produce 727 connected components and 440 shape-rule candidates. Label supervision claims exactly the expected 38 elements; only T1 remains flagged for its OCR-split label, while its solid bar passes the unchanged continuity test.
- Tributary Voronoi cells close exactly to the 828.0 m2 rectangular floor. At 8.75 kN/m2 and five storeys, column axial loads close to 36,225.0 kN.
- Visual acceptance: all eight red fixed-length bars lie on the black wall strokes; every extracted wall centre is within 0.5 m of the corrected independent reading. The one frozen VLM crop contains the T1 bar and printed 3500 x 400 callout.
- Status: accepted pending deterministic rerun, automated checks and Git commit.

## 2026-11-06 - Gate 7c: retain the NewPlan forest without selecting it

- Reused the exact FloorPlan1 seven feature maps, 80-tree forest, maximum depth 14, minimum leaf size 3, balanced class weights, seed 21345221 and 60,000-sample-per-class cap. The 46 m by 18 m frozen frame defines the prediction ROI with a 2 m margin; no forest output was used to tune that ROI.
- The forest trains only against the NewPlan classical record raster. It is therefore circular evidence: useful as a reproducibility witness, but prohibited from replacing or correcting the label-supervised record.
- Held-out F1 = 0.970087, OOB score = 0.971208 and full-ROI rule-target IoU = 0.557338. Density-family hand features carry 0.92673 of the total importance, confirming that the forest mostly re-learns the fixed morphology neighbourhoods.
- The 4.37 MB model, target/prediction masks, importance table and audit figure are retained. Visual review confirms that the prediction includes thin grid-line leakage despite strong held-out agreement, reinforcing the non-selection decision.
- The first combined regression command named the HW1 verifier incorrectly as `verify_hw1_outputs.py`; that path does not exist. HW2 and HW3 still passed, but the missing HW1 launch is not counted as a full regression gate. Correct the command to the repository's actual `HW1/scripts/verify_outputs.py` and rerun all three.
- Determinism: a second complete training run reproduced the model and all five derived artifacts byte for byte.
- Status: accepted pending the corrected three-homework regression run and Git commit; `selected` remains false.

## 2026-11-06 - Gate 7d: freeze the NewPlan dual-VLM execution

- NewPlan reuses the exact FloorPlan1 model IDs, pinned revisions, prompt Revision 2, parser Revision 2, seed, greedy decoding, 160-token limit, 0.85 confidence gate, contradiction guard and promotion ranking.
- The only evidence is the already frozen `NewPlan:T1:continuity` crop/question pair. Neither reader receives `NewPlan.png`, and the promoted copy cannot alter an element centre.
- Dimension comparison is made order-invariant for both plans: `3500x400` and `400x3500` are the same rectangular wall dimensions. A score still fails if either number differs; the FloorPlan1 comparison and promoted record must remain byte-identical after this general implementation change.
- Baseline non-regression: replaying the updated general scorer left the FloorPlan1 comparison and promoted record byte-identical.
- SmolVLM2 returned prose plus invalid JSON, used the wrong `size` key, read 3300 mm and reported confidence 0. Its verbatim output is retained; T1 remains unresolved for this reader.
- LLaVA-OneVision returned a fenced five-key object with quoted boolean/decimal values. The already accepted parser normalization removed the single fence and coerced those two literal types without changing values. It read `T1`, `400x3500`, continuous true and confidence 0.9, agreeing with the frozen independent reference.
- Promotion: LLaVA resolves 1/1 and breaks 0; SmolVLM2 resolves 0/1 and breaks 0. LLaVA is promoted, T1's classical audit flag is cleared and all 38 extracted centres remain exactly equal to the classical record.
- Status: Gate 7d accepted pending score-only reproducibility, automated checks and Git commit. No human answer was substituted.

## 2026-11-06 - Gate 7e: NewPlan hand CM/CR and VLM consequence

- Reused the FloorPlan1 material and mechanics unchanged: E = 30 GPa, G = 12.5 GPa, H = 3 m, shear factor 5/6 and the same bending-plus-shear wall flexibility. The rectangular 46 m by 18 m floor gives area 828.0 m2, CM = (23.0, 9.0) m, single-floor mass 738.532 t and polar area 168,360 m4.
- The promoted eight-wall extraction gives CR = (23.00745, 9.945907) m and eccentricity 0.945937 m. The frozen independent wall centres give CR = (23.0, 9.918072) m and eccentricity 0.918072 m; the extracted/reference CR difference is 0.028815 m.
- The classical T1 continuity field was already true: its flag concerned the split OCR label, not a broken wall. Therefore unresolved SmolVLM2 and resolving LLaVA produce the same eight active walls and the same structural result. The useful VLM consequence here is audit certainty, not a change in stiffness topology.
- Under the fixed 1000 kN +x push at CM, the CR torque is 945.907 kN m. Wall forces close to 1000 kN in x, zero in y and the applied torque; a unit push through CR has zero rotation to saved precision.
- Bending-only eccentricity exceeds the shear-included value, so retaining the assignment's shear term remains material. The single-storey reference periods are Tx = 0.07077 s, Ty = 0.05504 s and Ttheta = 0.06752 s.
- Visual review accepts the extracted wall layout, nearly coincident extracted/reference CR markers and the push-plus-turn force chart.
- Status: accepted pending byte-for-byte rerun, automated checks and Git commit.
