# Homework 3 - Floor-Plan Reading and Rigid-Diaphragm Stiffness

This folder is the Homework 3 package for **Shuhan He / 何抒翰**, student ID **21345221**. The assigned input is `assets/FloorPlan1.png`; `assets/NewPlan.png` and one independently sourced public plan are the fixed-pipeline holdouts.

## Required pipeline order

1. Run the fixed L07 classical reader: threshold, morphology, components, shape rules, grid and tributary areas.
2. Let printed labels supervise one element each and write an immutable classical record with an audit block.
3. Train and archive the per-pixel random forest, but never promote its extraction.
4. Ask two different VLMs only the frozen flags, using identical crops, questions and order.
5. Score both runs against an independent reference, promote the non-regressing winner and retain the loser as a witness.
6. Compute CM and CR by hand, reproduce them with an OpenSees rigid diaphragm, then modify the wall system so CR approaches CM.
7. Re-run the identical frozen pipeline on `NewPlan.png` and a public found plan without retuning.

## Repository rules

- `assignment/`, `assets/`, `../HW1/` and `../HW2/` are immutable inputs or accepted prior work.
- Coordinates are in metres; forces are in kN; stiffness is in consistent kN-m units.
- The original classical JSON is never overwritten. Every VLM works on its own copy.
- Every stage is recorded in `briefings/iteration_log-HW3.md` before it is committed.
- Generated results must be reproducible from scripts and pass `scripts/verify_hw3_outputs.py`.

## Primary deliverables

- `report/21345221-report-HW3.md`
- `output/pdf/21345221-report-HW3.pdf`
- three plan records, independent references, flag manifests and two VLM outputs per plan
- archived random forests and feature-importance tables
- hand and OpenSees stiffness results, wall-force tables, modal checks and the CM=CR modification

The source assignment and asset hashes are frozen in `config/input_manifest-HW3.json`.

## Reproduce and verify

From the cumulative repository root, use the main environment for deterministic extraction, scoring and report generation. Real VLM responses are already retained, so the routine commands score those frozen responses instead of downloading models again.

```powershell
& .\.venv\Scripts\python.exe HW3\scripts\run_classical_pipeline.py
& .\.venv\Scripts\python.exe HW3\scripts\train_random_forest.py
& .\.venv\Scripts\python.exe HW3\scripts\run_dual_vlm_flags.py --score-only
& .\.venv\Scripts\python.exe HW3\scripts\run_hand_structural.py
powershell -File HW3\scripts\run_opensees_py312.ps1
& .\.venv\Scripts\python.exe HW3\scripts\run_modified_structural.py
& .\.venv\Scripts\python.exe HW3\scripts\run_newplan_classical.py
& .\.venv\Scripts\python.exe HW3\scripts\run_newplan_dual_vlm_flags.py --score-only
& .\.venv\Scripts\python.exe HW3\scripts\run_mlstruct_classical.py
& .\.venv\Scripts\python.exe HW3\scripts\run_mlstruct_dual_vlm_flags.py --score-only
& .\.venv\Scripts\python.exe HW3\scripts\run_mlstruct_reference_structural.py
& .\.venv\Scripts\python.exe HW3\scripts\build_hw3_report.py
& .\.venv\Scripts\python.exe HW3\scripts\render_hw3_report_pdf.py
& .\.venv\Scripts\python.exe HW3\scripts\verify_hw3_outputs.py
```

OpenSees is isolated in `.opensees-py312` because the Windows extension requires its Python 3.12 ABI. The complete command list, assumptions, results and limitations are in `report/21345221-report-HW3.md`; the submission PDF is `output/pdf/21345221-report-HW3.pdf`.
