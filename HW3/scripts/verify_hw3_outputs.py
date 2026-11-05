from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
PLAN = ROOT / "output" / "FloorPlan1"


def load_json(path: Path) -> object:
    if not path.exists():
        raise AssertionError(f"Missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def require(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise AssertionError(f"Missing or empty file: {path}")


def verify_frozen_inputs() -> None:
    manifest = load_json(ROOT / "config" / "input_manifest-HW3.json")
    assert manifest["assigned_plan"] == "HW3/assets/FloorPlan1.png"
    for item in manifest["files"]:
        path = REPO / item["path"]
        require(path)
        assert path.stat().st_size == item["bytes"], path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"], path


def verify_classical() -> None:
    params = load_json(ROOT / "config" / "classical-params-HW3.json")
    cfg = params["classical"]
    assert (cfg["grey_threshold"], cfg["diagonal_kernel_px"], cfg["horizontal_open_px"], cfg["vertical_open_px"]) == (225, 11, 17, 21)

    record = load_json(PLAN / "results" / "record-classical-label-supervised-HW3.json")
    elements = record["elements"]
    labels = [item["labels"][0] for item in elements]
    assert len(elements) == 18 and len(set(labels)) == 18
    assert sum(item["class"] == "column" for item in elements) == 7
    assert sum(item["class"] == "wall" for item in elements) == 11
    assert record["method"]["forest_selected"] is False
    assert record["method"]["vlm_used_for_coordinates"] is False
    assert abs(record["floor"]["outline_area_m2"] - 189.33) < 1e-6
    assert abs(record["floor"]["effective_area_m2"] - 180.1285) < 1e-6
    assert abs(record["checks"]["sum_axial_kN"] - 7880.622) < 1e-6
    assert 125 < record["scale_px_per_m"]["x"] < 135
    assert 125 < record["scale_px_per_m"]["y"] < 135

    score = load_json(PLAN / "results" / "score-sheet-classical-HW3.json")
    assert score["named_elements_expected"] == score["named_elements_found"] == 18
    assert score["labels_original_pass"] == 18
    assert score["labels_long_lines_removed_pass"] == 18
    assert score["broken_or_low_support_named_elements"] == 4

    flags = load_json(PLAN / "results" / "flags-manifest-HW3.json")
    assert flags["confidence_threshold"] == 0.85
    assert [item["label"] for item in flags["flags"]] == ["T3", "T4", "T10", "T11"]
    questions = set()
    for item in flags["flags"]:
        crop = ROOT / item["crop"]
        require(crop)
        assert hashlib.sha256(crop.read_bytes()).hexdigest() == item["crop_sha256"]
        assert item["question"] not in questions
        questions.add(item["question"])

    reference = load_json(ROOT / "config" / "reference-manual-FloorPlan1-HW3.json")
    assert "before either VLM run" in reference["route"]
    for item in flags["flags"]:
        expected = reference["walls"][item["label"]]
        assert expected["continuous"] is True
        assert expected["size_mm"] == item["answer_schema"]["size_mm"]

    for name in (
        "hw3_01_classical_morphology.png",
        "hw3_02_connected_components.png",
        "hw3_03_label_supervision.png",
        "hw3_04_label_supervised_record.png",
    ):
        require(PLAN / "figures" / name)
    for name in (
        "floorplan1_threshold.png",
        "floorplan1_diagonal_support.png",
        "floorplan1_horizontal_open17.png",
        "floorplan1_vertical_open21.png",
        "floorplan1_long_lines_removed.png",
    ):
        require(PLAN / "intermediate" / name)


def verify_optional_later_stages() -> None:
    forest = PLAN / "results" / "random-forest-summary-HW3.json"
    if forest.exists():
        payload = load_json(forest)
        assert payload["selected"] is False
        assert abs(sum(payload["feature_importance"].values()) - 1.0) < 1e-6
        require(ROOT / payload["model_path"])


def main() -> None:
    verify_frozen_inputs()
    verify_classical()
    verify_optional_later_stages()
    print("HW3 verification passed: frozen inputs, 18 labels, four audited flags, identical-crop hashes and independent reference are present.")


if __name__ == "__main__":
    main()
