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


def verify_task8_frozen_sources() -> None:
    config_path = ROOT / "config" / "generalization-plans-HW3.json"
    if not config_path.exists():
        return
    config = load_json(config_path)
    assert config["locked_classical_parameters"] == "config/classical-params-HW3.json"
    new_plan = config["plans"]["NewPlan"]
    found = config["plans"]["MLStructFP966"]
    frozen = [
        (ROOT / new_plan["source"], new_plan["sha256"]),
        (ROOT / found["source"], found["sha256"]),
        (ROOT / found["annotation_source"], found["annotation_sha256"]),
        (ROOT / found["license"], found["license_sha256"]),
    ]
    for path, expected in frozen:
        require(path)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, path
        assert path.stat().st_size < 6_000_000, path

    assert len(new_plan["columns"]) == 30
    assert len(new_plan["walls"]) == 8
    assert new_plan["printed_column_size_mm"] == [800, 800]
    assert set(new_plan["printed_wall_sizes_mm"]) == {
        f"T{index}" for index in range(1, 9)
    }

    new_reference = load_json(
        ROOT / "config" / "reference-manual-NewPlan-HW3.json"
    )
    assert new_reference["built_before_vlm"] is True
    assert new_reference["source_sha256"] == new_plan["sha256"]
    assert new_reference["floor"]["area_m2"] == 828.0
    assert new_reference["floor"]["centroid_m"] == [23.0, 9.0]
    assert len(new_reference["columns"]) == 30
    assert len(new_reference["walls"]) == 8
    assert all(item["continuous"] for item in new_reference["walls"].values())

    found_reference = load_json(
        ROOT / "config" / "reference-dataset-MLStructFP966-HW3.json"
    )
    assert found_reference["built_before_vlm"] is True
    assert found_reference["dataset_floor_id"] == 966
    assert found_reference["scale_px_per_m"] == 366.506
    assert found_reference["wall_group_count"] == 20
    assert len(found_reference["wall_rectangles"]) == 28
    assert found_reference["printed_element_size_labels"] == 0
    assert abs(found_reference["floor"]["area_m2"] - 383.86362) < 1e-8
    assert any(
        abs(item["angle_deg"] - 8.5) < 1e-9
        for item in found_reference["wall_rectangles"]
    )

    flag_reference_path = (
        ROOT / "config" / "reference-flags-MLStructFP966-HW3.json"
    )
    if flag_reference_path.exists():
        flag_reference = load_json(flag_reference_path)
        assert flag_reference["built_before_vlm"] is True
        assert flag_reference["source_sha256"] == found["sha256"]
        assert flag_reference["candidate_count"] == 15
        assert flag_reference["can_supply_structural_geometry"] is False
        assert list(flag_reference["walls"]) == [
            f"U{index:03d}" for index in range(1, 16)
        ]
        assert all(
            item["size_mm"] == "NOT_VISIBLE"
            and item["continuous"] is True
            and item["printed_label_present"] is False
            and item["printed_complete_size_present"] is False
            and item["nearest_reference_distance_m"] < 0.25
            for item in flag_reference["walls"].values()
        )


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


def verify_newplan_classical() -> None:
    plan_dir = ROOT / "output" / "NewPlan"
    record_path = (
        plan_dir / "results" / "record-classical-label-supervised-NewPlan-HW3.json"
    )
    if not record_path.exists():
        return
    record = load_json(record_path)
    elements = record["elements"]
    labels = [item["labels"][0] for item in elements]
    assert len(elements) == len(set(labels)) == 38
    assert sum(item["class"] == "column" for item in elements) == 30
    assert sum(item["class"] == "wall" for item in elements) == 8
    assert record["method"]["locked_to_floorplan1_parameters"] is True
    assert record["method"]["forest_selected"] is False
    assert record["method"]["vlm_used_for_coordinates"] is False
    assert record["floor"]["outline_area_m2"] == 828.0
    assert record["checks"]["sum_tributary_areas_m2"] == 828.0
    assert record["checks"]["sum_axial_kN"] == 36225.0
    assert record["checks"]["floor_total_axial_kN"] == 36225.0
    assert 110 < record["scale_px_per_m"]["x"] < 116
    assert 110 < record["scale_px_per_m"]["y"] < 116

    score = load_json(
        plan_dir / "results" / "score-sheet-classical-NewPlan-HW3.json"
    )
    assert score["named_elements_expected"] == score["named_elements_found"] == 38
    assert score["columns_found"] == 30 and score["walls_found"] == 8
    assert score["split_label_reconstructions"] == ["T1"]
    assert score["broken_or_low_support_named_elements"] == 1
    assert score["flag_labels"] == ["T1"]

    flags = load_json(plan_dir / "results" / "flags-manifest-NewPlan-HW3.json")
    assert flags["crops_frozen_before_vlm"] is True
    assert flags["confidence_threshold"] == 0.85
    assert flags["question_order"] == ["NewPlan:T1:continuity"]
    assert len(flags["flags"]) == 1
    flag = flags["flags"][0]
    crop = ROOT / flag["crop"]
    require(crop)
    assert hashlib.sha256(crop.read_bytes()).hexdigest() == flag["crop_sha256"]
    assert flag["answer_schema"]["size_mm"] == "400x3500"

    reference = load_json(ROOT / "config" / "reference-manual-NewPlan-HW3.json")
    reference_walls = reference["walls"]
    walls = {
        item["labels"][0]: item for item in elements if item["class"] == "wall"
    }
    assert walls["T1"]["audit"]["label_reconstructed"] is True
    assert walls["T1"]["audit"]["flags"] == ["misread_label"]
    for label, wall in walls.items():
        expected = reference_walls[label]
        assert f"{wall['size_mm'][0]}x{wall['size_mm'][1]}" == expected["size_mm"]
        assert wall["continuous"] is expected["continuous"]
        assert abs(wall["centre_m"][0] - expected["centre_m"][0]) < 0.5
        assert abs(wall["centre_m"][1] - expected["centre_m"][1]) < 0.5

    for name in (
        "hw3_08_newplan_classical_morphology.png",
        "hw3_09_newplan_connected_components.png",
        "hw3_10_newplan_label_supervision.png",
        "hw3_11_newplan_label_supervised_record.png",
    ):
        require(plan_dir / "figures" / name)
    for name in (
        "newplan-threshold.png",
        "newplan-diagonal-support.png",
        "newplan-horizontal-open17.png",
        "newplan-vertical-open21.png",
        "newplan-long-lines-removed.png",
    ):
        require(plan_dir / "intermediate" / name)


def verify_newplan_forest() -> None:
    plan_dir = ROOT / "output" / "NewPlan"
    summary_path = plan_dir / "results" / "random-forest-summary-NewPlan-HW3.json"
    if not summary_path.exists():
        return
    summary = load_json(summary_path)
    assert summary["plan"] == "NewPlan"
    assert summary["selected"] is False
    assert summary["locked_to_floorplan1_features_and_hyperparameters"] is True
    assert summary["random_seed"] == 21345221
    assert summary["roi_px"] == [1402, 1163, 7132, 3634]
    assert summary["training_samples"] == 96000
    assert summary["test_samples"] == 24000
    assert summary["test_f1"] > 0.95
    assert summary["oob_score"] > 0.95
    assert 0.5 < summary["rule_target_iou"] < 0.7
    assert abs(sum(summary["feature_importance"].values()) - 1.0) < 1e-6
    hand_features = sum(
        value
        for name, value in summary["feature_importance"].items()
        if "density" in name
    )
    assert hand_features > 0.90
    model = ROOT / summary["model_path"]
    require(model)
    assert model.stat().st_size < 6_000_000
    require(plan_dir / "results" / "random-forest-feature-importance-NewPlan-HW3.csv")
    require(plan_dir / "intermediate" / "forest-rule-target-NewPlan-HW3.png")
    require(plan_dir / "intermediate" / "forest-prediction-NewPlan-HW3.png")
    require(plan_dir / "figures" / "hw3_12_newplan_random_forest_never_chosen.png")


def verify_newplan_vlm() -> None:
    plan_dir = ROOT / "output" / "NewPlan"
    comparison_path = plan_dir / "results" / "vlm-comparison-NewPlan-HW3.json"
    if not comparison_path.exists():
        return
    manifest = load_json(
        plan_dir / "results" / "flags-manifest-NewPlan-HW3.json"
    )
    expected_order = [item["flag_id"] for item in manifest["flags"]]
    result_names = (
        "vlm-smolvlm2-256m-NewPlan-HW3.json",
        "vlm-llava-onevision-0.5b-NewPlan-HW3.json",
    )
    results = [load_json(plan_dir / "results" / name) for name in result_names]
    assert [item["model_key"] for item in results] == [
        "smolvlm2-256m",
        "llava-onevision-0.5b",
    ]
    assert [item["revision"] for item in results] == [
        "067788b187b95ebe7b2e040b3e4299e342e5b8fd",
        "74dd0bf867a4cda7950c17663794267c60cf4b40",
    ]
    for result in results:
        assert result["plan"] == "NewPlan"
        assert result["prompt_version"] == 2
        assert result["parser_version"] == 2
        assert result["whole_plan_seen"] is False
        assert result["coordinates_editable_by_vlm"] is False
        assert result["question_order"] == expected_order
        assert len(result["answers"]) == 1
        answer = result["answers"][0]
        flag = manifest["flags"][0]
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
    assert scores["smolvlm2-256m"]["flags_unresolved"] == 1
    assert (scores["llava-onevision-0.5b"]["flags_resolved"], scores["llava-onevision-0.5b"]["flags_broken"]) == (1, 0)
    assert scores["llava-onevision-0.5b"]["mean_size_error_mm"] == 0.0
    assert scores["llava-onevision-0.5b"]["mean_reported_confidence"] == 0.9

    classical = load_json(
        plan_dir / "results" / "record-classical-label-supervised-NewPlan-HW3.json"
    )
    promoted = load_json(
        plan_dir / "results" / "record-promoted-NewPlan-HW3.json"
    )
    assert promoted["source"]["promoted_vlm"] == comparison["promoted_model_id"]
    classical_elements = {item["labels"][0]: item for item in classical["elements"]}
    promoted_elements = {item["labels"][0]: item for item in promoted["elements"]}
    assert classical_elements.keys() == promoted_elements.keys()
    for label in classical_elements:
        assert promoted_elements[label]["centre_m"] == classical_elements[label]["centre_m"]
    t1 = promoted_elements["T1"]
    assert t1["continuous"] is True
    assert t1["audit"]["flags"] == []
    assert t1["audit"]["vlm_resolution"]["coordinates_changed"] is False


def verify_newplan_structural() -> None:
    plan_dir = ROOT / "output" / "NewPlan"
    hand_path = plan_dir / "results" / "hand-structural-NewPlan-HW3.json"
    if not hand_path.exists():
        return
    hand = load_json(hand_path)
    assert hand["plan"] == "NewPlan"
    assert hand["mechanics"]["locked_to_floorplan1_mechanics"] is True
    mass = hand["mass"]
    assert mass["effective_area_m2"] == 828.0
    assert mass["cm_m"] == [23.0, 9.0]
    assert abs(mass["polar_area_about_cm_m4"] - 168360.0) < 1e-9

    promoted = hand["promoted_shear_included"]
    reference = hand["independent_reference_shear"]
    expected_walls = [f"T{index}" for index in range(1, 9)]
    assert promoted["included_walls"] == expected_walls
    assert reference["included_walls"] == expected_walls
    assert abs(promoted["cr_m"][0] - 23.00745) < 1e-8
    assert abs(promoted["cr_m"][1] - 9.945907229) < 1e-8
    assert abs(promoted["eccentricity_m"] - 0.945936567) < 1e-8
    assert abs(reference["cr_m"][0] - 23.0) < 1e-8
    assert abs(reference["cr_m"][1] - 9.918072289) < 1e-8
    assert abs(reference["eccentricity_m"] - 0.918072289) < 1e-8
    assert hand["promoted_bending_only"]["eccentricity_m"] > promoted["eccentricity_m"]

    for row_index, row in enumerate(promoted["K_at_cm"]):
        for column_index, value in enumerate(row):
            assert abs(value - promoted["K_at_cm"][column_index][row_index]) < 1e-6
    for model_key in ("smolvlm2-256m", "llava-onevision-0.5b"):
        consequence = hand["vlm_structural_consequence"][model_key]
        assert consequence["cr_m"] == promoted["cr_m"]
        assert all(consequence["continuity_states"].values())
        assert abs(consequence["shift_norm_from_reference_m"] - 0.02881469) < 1e-8
    assert max(abs(value) for value in hand["checks"].values()) < 1e-6
    design = hand["design_load_and_equilibrium"]
    assert abs(design["equilibrium_sum_fx_kN"] - 1000.0) < 1e-6
    assert abs(design["equilibrium_sum_fy_kN"]) < 1e-6
    assert abs(
        design["equilibrium_moment_about_cr_kNm"]
        - design["torque_about_cr_kNm"]
    ) < 1e-6

    csv_path = plan_dir / "results" / "wall-stiffness-and-forces-NewPlan-HW3.csv"
    require(csv_path)
    with csv_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 8
    assert [row["id"] for row in rows] == expected_walls
    require(plan_dir / "figures" / "hw3_13_newplan_cm_cr_and_wall_forces.png")


def verify_mlstruct_classical() -> None:
    plan_dir = ROOT / "output" / "MLStructFP966"
    record_path = (
        plan_dir
        / "results"
        / "record-classical-label-supervised-MLStructFP966-HW3.json"
    )
    if not record_path.exists():
        return
    record = load_json(record_path)
    assert record["elements"] == []
    assert len(record["unclaimed_candidates"]) == 15
    assert record["floor"]["pipeline_geometry"] is None
    assert record["method"]["locked_to_floorplan1_parameters"] is True
    assert record["method"]["forest_selected"] is False
    assert record["method"]["vlm_used_for_coordinates"] is False
    assert record["method"]["reference_used_to_claim_elements"] is False
    assert "failed safely" in record["generalization_status"]
    candidate_ids = [item["candidate_id"] for item in record["unclaimed_candidates"]]
    assert candidate_ids == [f"U{index:03d}" for index in range(1, 16)]
    assert all(item["claimed_as_element"] is False for item in record["unclaimed_candidates"])
    assert all(item["nearest_reference_distance_m"] < 0.25 for item in record["unclaimed_candidates"])

    score = load_json(
        plan_dir / "results" / "score-sheet-classical-MLStructFP966-HW3.json"
    )
    assert score["image_shape_px"] == [9600, 6784]
    assert score["fixed_parameters"] == {
        "grey_threshold": 225,
        "diagonal_kernel_px": 11,
        "horizontal_open_px": 17,
        "vertical_open_px": 21,
        "shape_ratio_min": 4.0,
        "ocr_acceptance": 0.85,
    }
    assert score["ocr_detections_original"] == 1
    assert score["ocr_detections_long_lines_removed"] == 0
    assert score["ocr_above_threshold"] == 0
    assert score["printed_element_size_labels_reference"] == 0
    assert score["connected_components_kept"] == 20
    assert score["shape_rule_elongated_candidates"] == 15
    assert score["label_claimed_elements"] == 0
    assert score["unclaimed_flags"] == 15
    assert score["audit_candidate_matches_within_0_25m"] == 15
    assert score["audit_unique_reference_rectangles_hit"] == 14
    assert score["audit_reference_rectangles_total"] == 28
    assert score["audit_unique_reference_recall"] == 0.5
    assert score["audit_duplicate_candidate_count"] == 1
    assert score["structural_record_available"] is False

    manifest = load_json(
        plan_dir / "results" / "flags-manifest-MLStructFP966-HW3.json"
    )
    assert manifest["crops_frozen_before_vlm"] is True
    assert manifest["whole_plan_crop_prohibited"] is True
    assert manifest["confidence_threshold"] == 0.85
    assert len(manifest["flags"]) == 15
    assert manifest["question_order"] == [item["flag_id"] for item in manifest["flags"]]
    seen_questions = set()
    for item in manifest["flags"]:
        crop = ROOT / item["crop"]
        require(crop)
        assert hashlib.sha256(crop.read_bytes()).hexdigest() == item["crop_sha256"]
        x0, y0, x1, y1 = item["crop_box_px"]
        assert 0 < x1 - x0 < 1200
        assert 0 < y1 - y0 < 1600
        assert item["answer_schema"]["size_mm"] == "NOT_VISIBLE"
        assert item["question"] not in seen_questions
        seen_questions.add(item["question"])

    for name in (
        "hw3_14_mlstruct_classical_morphology.png",
        "hw3_15_mlstruct_connected_components.png",
        "hw3_16_mlstruct_label_supervision_failure.png",
        "hw3_17_mlstruct_reference_comparison.png",
    ):
        require(plan_dir / "figures" / name)
    for name in (
        "mlstruct-composited-white.png",
        "mlstruct-threshold.png",
        "mlstruct-diagonal-support.png",
        "mlstruct-horizontal-open17.png",
        "mlstruct-vertical-open21.png",
        "mlstruct-long-lines-removed.png",
    ):
        require(plan_dir / "intermediate" / name)


def verify_mlstruct_forest() -> None:
    plan_dir = ROOT / "output" / "MLStructFP966"
    summary_path = (
        plan_dir / "results" / "random-forest-summary-MLStructFP966-HW3.json"
    )
    if not summary_path.exists():
        return
    summary = load_json(summary_path)
    assert summary["plan"] == "MLStructFP966"
    assert summary["selected"] is False
    assert summary["locked_to_floorplan1_features_and_hyperparameters"] is True
    assert summary["public_annotation_used_for_training"] is False
    assert summary["random_seed"] == 21345221
    assert summary["training_samples"] == 96000
    assert summary["test_samples"] == 24000
    assert summary["oob_score"] == 0.746687
    assert summary["test_f1"] == 0.674299
    assert summary["rule_target_iou"] == 0.221414
    assert abs(sum(summary["feature_importance"].values()) - 1.0) < 1e-6
    assert summary["feature_importance"]["ink_density_45"] > 0.69
    assert summary["feature_importance"]["stroke_45deg_density_21"] < 0.001
    model = ROOT / summary["model_path"]
    require(model)
    assert model.stat().st_size < 6_000_000
    require(
        plan_dir
        / "results"
        / "random-forest-feature-importance-MLStructFP966-HW3.csv"
    )
    require(
        plan_dir / "intermediate" / "forest-rule-target-MLStructFP966-HW3.png"
    )
    require(
        plan_dir / "intermediate" / "forest-prediction-MLStructFP966-HW3.png"
    )
    require(
        plan_dir / "figures" / "hw3_18_mlstruct_random_forest_never_chosen.png"
    )


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

    opensees_path = PLAN / "results" / "opensees-FloorPlan1-HW3.json"
    if opensees_path.exists():
        model = load_json(opensees_path)
        assert model["runtime"]["python"].startswith("3.12")
        assert model["runtime"]["opensees"].startswith("3.8")
        description = model["model"]
        assert description["element"] == "ElasticTimoshenkoBeam"
        assert description["rigid_diaphragm"].startswith("rigidDiaphragm(3")
        assert description["constraint_handler"] == "Transformation"
        assert description["included_walls"] == [
            "T1", "T2", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11"
        ]
        comparison = model["comparison_to_hand"]
        assert comparison["cr_difference_norm_m"] < 1e-6
        assert comparison["max_abs_K_relative_difference"] < 1e-6
        assert abs(comparison["unit_push_through_cr_rotation_rad"]) < 1e-12
        assert abs(comparison["design_wall_sum_x_kN"] - 1000.0) < 1e-5
        assert abs(comparison["design_cross_wall_sum_y_kN"]) < 1e-5
        assert comparison["max_wall_force_additivity_residual_kN"] < 1e-9
        hand_rows = {
            row["id"]: row
            for row in csv.DictReader(
                (PLAN / "results" / "wall-stiffness-and-forces-FloorPlan1-HW3.csv").open(
                    encoding="utf-8", newline=""
                )
            )
        }
        model_forces = model["design_cases"]["total_at_cm"]["wall_forces"]
        for wall_id in description["included_walls"]:
            assert abs(
                model_forces[wall_id]["strong_top_shear_kN"]
                - float(hand_rows[wall_id]["force_total_kN"])
            ) < 1e-5

    modified_hand_path = PLAN / "results" / "modified-hand-FloorPlan1-HW3.json"
    if modified_hand_path.exists():
        modified = load_json(modified_hand_path)
        assert modified["mass_layout_changed"] is False
        assert modified["centreline_or_column_moved"] is False
        assert modified["modifications"]["T3"]["new_thickness_m"] == 0.309
        assert modified["modifications"]["T3"]["new_continuous"] is True
        assert modified["modifications"]["T7"]["new_thickness_m"] == 0.365
        before_after = modified["before_after"]
        assert before_after["eccentricity_before_m"] > 1.1
        assert before_after["eccentricity_after_m"] < 0.001
        assert before_after["eccentricity_reduction_percent"] > 99.9
        assert before_after["baseline_most_torsion_amplified_wall"] == "T2"
        assert abs(before_after["baseline_max_amplification_ratio"] - 1.326010555) < 1e-8
        assert before_after["period_ratio_Ttheta_over_Tx_after"] < before_after["period_ratio_Ttheta_over_Tx_before"]
        assert max(abs(value) for value in modified["checks"].values()) < 0.001
        require(PLAN / "results" / "modified-wall-forces-FloorPlan1-HW3.csv")
        require(PLAN / "figures" / "hw3_07_modification_before_after.png")

    modified_ops_path = PLAN / "results" / "opensees-modified-FloorPlan1-HW3.json"
    if modified_ops_path.exists():
        modified_ops = load_json(modified_ops_path)
        assert modified_ops["layout"] == "modified"
        assert modified_ops["model"]["included_walls"] == [
            "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11"
        ]
        comparison = modified_ops["comparison_to_hand"]
        assert comparison["cr_difference_norm_m"] < 1e-6
        assert comparison["max_abs_K_relative_difference"] < 1e-4
        assert abs(comparison["unit_push_through_cr_rotation_rad"]) < 1e-12
        assert abs(comparison["design_wall_sum_x_kN"] - 1000.0) < 1e-5
        assert abs(comparison["design_cross_wall_sum_y_kN"]) < 1e-5
        assert comparison["max_wall_force_additivity_residual_kN"] < 1e-9


def main() -> None:
    verify_frozen_inputs()
    verify_task8_frozen_sources()
    verify_classical()
    verify_newplan_classical()
    verify_newplan_forest()
    verify_newplan_vlm()
    verify_newplan_structural()
    verify_mlstruct_classical()
    verify_mlstruct_forest()
    verify_optional_later_stages()
    print("HW3 verification passed: frozen inputs, classical/forest gates, identical dual-VLM evidence, syntax audit and promoted-copy invariants are present.")


if __name__ == "__main__":
    main()
