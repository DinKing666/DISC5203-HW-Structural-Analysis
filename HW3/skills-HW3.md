# Reusable HW3 procedures

## Read a plan without delegating measurement to a VLM

1. Apply the frozen grey threshold and morphology kernels.
2. Extract components and bars; measure their geometry from pixels.
3. Read text on both the original crop and a long-line-removed copy.
4. Let one label claim one bar only within its thickness class.
5. Record uncertain label, continuity, completion and unclaimed-bar cases as flags.
6. Freeze one crop and one checkable question per flag.
7. Use VLM answers only for printed text and continuity decisions; geometry remains rule-derived.

## Gate two VLM runs

1. Create the independent reference before opening either result.
2. Verify both runs used byte-identical crops and question manifests.
3. Accept only schema-valid answers above the frozen threshold.
4. Count agreement with the reference as resolved and disagreement as broken.
5. Reject a candidate that regresses a flag the other candidate clears.
6. Promote the eligible run with more resolved flags and keep both raw outputs.

## Check a rigid-diaphragm calculation

1. Compute each wall's bending-plus-shear cantilever stiffness by hand.
2. Assemble the 3x3 floor matrix about CM and calculate CR and eccentricity.
3. Independently apply three unit loads in OpenSees and recover the flexibility matrix.
4. Require centimetre-level or better agreement between the two routes.
5. Check push through CR, push through CM, and push-plus-turn superposition.
6. Re-run the same checks after the wall-system modification.
