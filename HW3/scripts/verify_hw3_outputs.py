from __future__ import annotations

import hashlib
import json
import csv
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
        assert payload["test_f1"] > 0.95
        hand_features = sum(
            value for name, value in payload["feature_importance"].items()
            if "density" in name
        )
        assert hand_features > 0.75
        model = ROOT / payload["model_path"]
        require(model)
        assert model.stat().st_size < 50_000_000
        require(PLAN / "figures" / "hw3_05_random_forest_never_chosen.png")
        require(PLAN / "intermediate" / "forest-prediction-HW3.png")

    comparison_path = PLAN / "results" / "vlm-comparison-FloorPlan1-HW3.json"
    if comparison_path.exists():
        result_names = (
            "vlm-smolvlm2-256m-FloorPlan1-HW3.json",
            "vlm-llava-onevision-0.5b-FloorPlan1-HW3.json",
        )
        manifest = load_json(PLAN / "results" / "flags-manifest-HW3.json")
        expected_order = [item["flag_id"] for item in manifest["flags"]]
        results = [load_json(PLAN / "results" / name) for name in result_names]
        assert [item["model_key"] for item in results] == [
            "smolvlm2-256m",
            "llava-onevision-0.5b",
        ]
        assert [item["revision"] for item in results] == [
            "067788b187b95ebe7b2e040b3e4299e342e5b8fd",
            "74dd0bf867a4cda7950c17663794267c60cf4b40",
        ]
        for result in results:
            assert result["prompt_version"] == 2
            assert result["parser_version"] == 2
            assert result["whole_plan_seen"] is False
            assert result["coordinates_editable_by_vlm"] is False
            assert result["question_order"] == expected_order
            assert len(result["answers"]) == 4
            for answer, flag in zip(result["answers"], manifest["flags"], strict=True):
                assert answer["flag_id"] == flag["flag_id"]
                assert answer["crop_sha256"] == flag["crop_sha256"]
                assert answer["question"] == flag["question"]
                assert answer["raw_answer"].strip()

        comparison = load_json(comparison_path)
        assert comparison["same_crops_questions_order"] is True
        assert comparison["gate_passed"] is True
        assert comparison["promoted_model_key"] == "llava-onevision-0.5b"
        scores = {item["model_key"]: item for item in comparison["models"]}
        assert (scores["smolvlm2-256m"]["flags_resolved"], scores["smolvlm2-256m"]["flags_broken"]) == (0, 0)
        assert (scores["llava-onevision-0.5b"]["flags_resolved"], scores["llava-onevision-0.5b"]["flags_broken"]) == (3, 0)
        assert scores["llava-onevision-0.5b"]["flags_unresolved"] == 1
        assert scores["llava-onevision-0.5b"]["mean_size_error_mm"] == 0.0
        t3 = next(
            item for item in scores["llava-onevision-0.5b"]["details"]
            if item["label"] == "T3"
        )
        assert t3["status"] == "unresolved" and t3["internal_contradiction"] is True

        classical = load_json(PLAN / "results" / "record-classical-label-supervised-HW3.json")
        promoted = load_json(PLAN / "results" / "record-promoted-FloorPlan1-HW3.json")
        assert promoted["source"]["promoted_vlm"] == comparison["promoted_model_id"]
        classical_elements = {
            item["labels"][0]: item for item in classical["elements"]
        }
        promoted_elements = {
            item["labels"][0]: item for item in promoted["elements"]
        }
        assert classical_elements.keys() == promoted_elements.keys()
        for label in classical_elements:
            assert promoted_elements[label]["centre_m"] == classical_elements[label]["centre_m"]
        assert promoted_elements["T3"]["continuous"] is False
        assert promoted_elements["T3"]["audit"]["flags"]
        for label in ("T4", "T10", "T11"):
            element = promoted_elements[label]
            assert element["audit"]["flags"] == []
            assert element["audit"]["vlm_resolution"]["coordinates_changed"] is False

        rejected = PLAN / "results" / "rejected"
        for name in (
            "vlm-comparison-FloorPlan1-prompt-v1-HW3.json",
            "vlm-comparison-FloorPlan1-prompt-v2-strict-parser-HW3.json",
        ):
            require(rejected / name)

    hand_path = PLAN / "results" / "hand-structural-FloorPlan1-HW3.json"
    if hand_path.exists():
        hand = load_json(hand_path)
        mass = hand["mass"]
        assert abs(mass["effective_area_m2"] - 180.1285) < 1e-9
        assert abs(mass["cm_m"][0] - 5.460450637) < 1e-8
        assert abs(mass["cm_m"][1] - 8.791522260) < 1e-8
        promoted = hand["promoted_shear_included"]
        reference = hand["independent_reference_shear"]
        assert "T3" not in promoted["included_walls"]
        assert "T3" in reference["included_walls"]
        assert abs(promoted["cr_m"][0] - 5.548844571) < 1e-8
        assert abs(promoted["cr_m"][1] - 9.894588860) < 1e-8
        assert abs(promoted["eccentricity_m"] - 1.106602643) < 1e-8
        assert abs(reference["cr_m"][1] - 8.980429046) < 1e-8
        assert hand["promoted_bending_only"]["eccentricity_m"] > promoted["eccentricity_m"] + 0.8
        for row_index, row in enumerate(promoted["K_at_cm"]):
            for column_index, value in enumerate(row):
                assert abs(value - promoted["K_at_cm"][column_index][row_index]) < 1e-6
        checks = hand["checks"]
        assert max(abs(value) for value in checks.values()) < 1e-6
        assert hand["modal_properties"]["coupled_modes"][0]["dominant_character"] == "torsion"
        assert hand["modal_properties"]["period_ratio_Ttheta_over_Tx"] > 1.0
        for model in ("smolvlm2-256m", "llava-onevision-0.5b"):
            consequence = hand["vlm_structural_consequence"][model]
            assert abs(consequence["shift_norm_from_reference_m"] - 0.914159814) < 1e-8
        csv_path = PLAN / "results" / "wall-stiffness-and-forces-FloorPlan1-HW3.csv"
        require(csv_path)
        with csv_path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 11
        assert next(row for row in rows if row["id"] == "T3")["continuous"] == "False"
        require(PLAN / "figures" / "hw3_06_cm_cr_and_wall_forces.png")


def main() -> None:
    verify_frozen_inputs()
    verify_classical()
    verify_optional_later_stages()
    print("HW3 verification passed: frozen inputs, classical/forest gates, identical dual-VLM evidence, syntax audit and promoted-copy invariants are present.")


if __name__ == "__main__":
    main()
