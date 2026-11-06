"""Run the frozen FloorPlan1 dual-VLM gate on NewPlan's frozen crop."""

from __future__ import annotations

import copy

import run_dual_vlm_flags as gate


gate.PLAN = "NewPlan"
gate.MANIFEST_PATH = (
    gate.ROOT / "HW3/output/NewPlan/results/flags-manifest-NewPlan-HW3.json"
)
gate.REFERENCE_PATH = gate.ROOT / "HW3/config/reference-manual-NewPlan-HW3.json"
gate.CLASSICAL_PATH = (
    gate.ROOT
    / "HW3/output/NewPlan/results/record-classical-label-supervised-NewPlan-HW3.json"
)
gate.RESULTS_DIR = gate.ROOT / "HW3/output/NewPlan/results"
gate.MODELS = copy.deepcopy(gate.MODELS)
gate.MODELS["smolvlm2-256m"]["result"] = (
    "vlm-smolvlm2-256m-NewPlan-HW3.json"
)
gate.MODELS["llava-onevision-0.5b"]["result"] = (
    "vlm-llava-onevision-0.5b-NewPlan-HW3.json"
)


if __name__ == "__main__":
    gate.main()
