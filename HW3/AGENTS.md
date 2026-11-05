# HW3 agent guidance

## Scope and invariants

- Work only inside `HW3/`, except for the cumulative root index and shared Git hook.
- Treat `assignment/`, `assets/`, `../HW1/` and `../HW2/` as read-only.
- Student ID 21345221 selects Floor Plan 1.
- Use kN and m throughout. Concrete uses E = 30 GPa, nu = 0.2, G = E/(2(1+nu)); storey height is 3 m and default wall thickness is 0.25 m.

## Extraction order

- Preserve the L09 order: classical rules, one-label/one-element supervision, kept-but-never-chosen forest, then VLM flag resolution.
- Never ask a VLM to extract the whole drawing or to move a vertex.
- Freeze crop bytes, question text, flag order, confidence threshold and scoring rule before either VLM run.
- Build the independent reference without consulting either VLM output.
- Never mutate the baseline record; write one copy per VLM and a separate promoted record.
- Lock classical parameters on Floor Plan 1 before evaluating NewPlan and the found plan.

## Structural checks

- Report the hand matrix before the OpenSees check.
- Include shear deformation for squat walls and retain the bending-only comparison.
- Use `ops.rigidDiaphragm(3, master, *slaves)` with Transformation constraints.
- Verify stiffness symmetry, flexibility inversion, hand/model CR agreement, force equilibrium, zero rotation for a push through CR and push-plus-turn superposition.
- Keep mass layout fixed when modifying the wall system; preserve columns, openings and the architectural grid.

## Iteration and Git

- Append every material decision, failed assumption, correction and acceptance check to `briefings/iteration_log-HW3.md`.
- Commit only after the relevant verifier passes. Stage explicit paths and inspect the staged diff; do not use `git add .`.
- Keep generated evidence required by the assignment, including the losing VLM run and the forest.
- Do not push or rewrite published history without the student's explicit instruction.
