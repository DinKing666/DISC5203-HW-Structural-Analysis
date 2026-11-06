"""Freeze the MLStructFP966 crop-audit reference before either VLM runs."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "output/MLStructFP966/results/flags-manifest-MLStructFP966-HW3.json"
RECORD = (
    ROOT
    / "output/MLStructFP966/results/record-classical-label-supervised-MLStructFP966-HW3.json"
)
REFERENCE = ROOT / "config/reference-dataset-MLStructFP966-HW3.json"
DESTINATION = ROOT / "config/reference-flags-MLStructFP966-HW3.json"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    manifest = read_json(MANIFEST)
    record = read_json(RECORD)
    reference = read_json(REFERENCE)
    candidates = {
        item["candidate_id"]: item for item in record["unclaimed_candidates"]
    }
    walls = {}
    for flag in manifest["flags"]:
        candidate_id = flag["label"]
        candidate = candidates[candidate_id]
        walls[candidate_id] = {
            "internal_candidate_id": candidate_id,
            "size_mm": "NOT_VISIBLE",
            "continuous": candidate["nearest_reference_continuous"],
            "printed_label_present": False,
            "printed_complete_size_present": False,
            "nearest_dataset_reference": candidate["nearest_frozen_reference"],
            "nearest_reference_distance_m": candidate["nearest_reference_distance_m"],
            "reference_role": "continuity audit only; not an element identity or size",
        }
    payload = {
        "plan": "MLStructFP966",
        "built_before_vlm": True,
        "reference_route": (
            "Each committed classical candidate is paired to the nearest already-frozen "
            "MLStructFP rectangle within 0.25 m. The public annotation supplies continuity "
            "only; NOT_VISIBLE records that no complete size is printed in the crop."
        ),
        "source_reference": "config/reference-dataset-MLStructFP966-HW3.json",
        "source_sha256": reference["source_sha256"],
        "candidate_count": len(walls),
        "can_supply_structural_geometry": False,
        "walls": walls,
    }
    DESTINATION.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Frozen {len(walls)} MLStructFP966 crop-audit references before VLM")


if __name__ == "__main__":
    main()
