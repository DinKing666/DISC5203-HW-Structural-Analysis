"""Run the two frozen VLMs on MLStructFP966's 15 committed local crops."""

from __future__ import annotations

import argparse
import copy
from typing import Any

import run_dual_vlm_flags as gate


gate.PLAN = "MLStructFP966"
gate.PROMPT_VERSION = 3
gate.MANIFEST_PATH = (
    gate.ROOT
    / "HW3/output/MLStructFP966/results/flags-manifest-MLStructFP966-HW3.json"
)
gate.REFERENCE_PATH = (
    gate.ROOT / "HW3/config/reference-flags-MLStructFP966-HW3.json"
)
gate.CLASSICAL_PATH = (
    gate.ROOT
    / "HW3/output/MLStructFP966/results/record-classical-label-supervised-MLStructFP966-HW3.json"
)
gate.RESULTS_DIR = gate.ROOT / "HW3/output/MLStructFP966/results"
gate.MODELS = copy.deepcopy(gate.MODELS)
gate.MODELS["smolvlm2-256m"]["result"] = (
    "vlm-smolvlm2-256m-MLStructFP966-HW3.json"
)
gate.MODELS["llava-onevision-0.5b"]["result"] = (
    "vlm-llava-onevision-0.5b-MLStructFP966-HW3.json"
)


def crop_audit_prompt(flag: dict[str, Any]) -> str:
    return (
        f'{flag["question"]}\n'
        "Inspect only this crop. The U-number is an internal audit handle supplied in the "
        "question, not text printed in the drawing. NOT_VISIBLE is the required sentinel "
        "when no complete numeric size is printed. Return exactly one valid JSON object "
        "with exactly these five keys: label, size_mm, continuous, confidence, reason. "
        "Use the requested U-number for label and NOT_VISIBLE for size_mm; continuous must "
        "be your visual true/false judgment; confidence must be a non-placeholder decimal "
        "from 0 to 1; reason must be one short string. Output JSON only, with no markdown "
        "and no text or punctuation outside the closing brace."
    )


gate.frozen_prompt = crop_audit_prompt


def score_and_compare(manifest: dict[str, Any]) -> None:
    gate.refresh_parser_outputs()
    reference = gate.read_json(gate.REFERENCE_PATH)
    results = {
        key: gate.read_json(gate.RESULTS_DIR / spec["result"])
        for key, spec in gate.MODELS.items()
    }
    scores = [
        gate.score_model(results[key], manifest, reference) for key in gate.MODELS
    ]
    safe = [item for item in scores if item["flags_broken"] == 0]
    winner = (
        sorted(safe, key=gate.promotion_rank, reverse=True)[0] if safe else None
    )
    comparison = {
        "plan": gate.PLAN,
        "independent_reference": str(gate.REFERENCE_PATH.relative_to(gate.ROOT)).replace(
            "\\", "/"
        ),
        "confidence_threshold": manifest["confidence_threshold"],
        "same_crops_questions_order": True,
        "selection_rule": (
            "zero broken required; then more resolved; ties use lower mean size error, "
            "higher mean reported confidence, then deterministic model key"
        ),
        "models": scores,
        "best_safe_reader_key": winner["model_key"] if winner else None,
        "best_safe_reader_id": (
            results[winner["model_key"]]["model_id"] if winner else None
        ),
        "gate_passed": winner is not None,
        "structural_record_promoted": False,
        "structural_record_prohibition": (
            "The classical record has zero claimed elements and NOT_VISIBLE is not a "
            "wall size, so continuity answers cannot supply structural geometry."
        ),
    }
    gate.write_json(gate.comparison_path(), comparison)
    if winner:
        print(
            f"Best safe reader {winner['model_key']}: "
            f"{winner['flags_resolved']} resolved, {winner['flags_broken']} broken; "
            "no structural record promoted"
        )
    else:
        print("No reader passed the zero-broken gate; no structural record promoted")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["all", *gate.MODELS], default="all")
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()
    manifest = gate.read_json(gate.MANIFEST_PATH)
    if not args.score_only:
        gate.archive_rejected_prompt_outputs()
        keys = list(gate.MODELS) if args.model == "all" else [args.model]
        for key in keys:
            gate.run_model(key, manifest)
    result_paths = [gate.RESULTS_DIR / spec["result"] for spec in gate.MODELS.values()]
    if all(path.exists() for path in result_paths):
        score_and_compare(manifest)
    else:
        missing = ", ".join(path.name for path in result_paths if not path.exists())
        print(f"Scoring deferred until the remaining frozen output exists: {missing}")


if __name__ == "__main__":
    main()
