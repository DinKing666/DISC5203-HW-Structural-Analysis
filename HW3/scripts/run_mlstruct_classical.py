"""Run the locked FloorPlan1 classical reader on public MLStructFP floor 966."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from rapidocr import RapidOCR

from run_classical_pipeline import (
    connected_component_sheet,
    morphology,
    read_image,
    remove_long_lines,
    serialise_ocr,
    write_image,
)


ROOT = Path(__file__).resolve().parents[1]
PARAMS = ROOT / "config/classical-params-HW3.json"
GENERALISATION = ROOT / "config/generalization-plans-HW3.json"
REFERENCE = ROOT / "config/reference-dataset-MLStructFP966-HW3.json"
OUT = ROOT / "output/MLStructFP966"
FIGURES = OUT / "figures"
INTERMEDIATE = OUT / "intermediate"
RESULTS = OUT / "results"
CROPS = OUT / "flags/crops"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def alpha_to_white(source: Path) -> np.ndarray:
    rgba = read_image(source, cv2.IMREAD_UNCHANGED)
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise RuntimeError(f"Expected RGBA MLStructFP source: {source}")
    bgr = rgba[:, :, :3].copy()
    bgr[rgba[:, :, 3] == 0] = 255
    return bgr


def candidate_sheet(
    elongated: list[dict[str, Any]], reference: dict[str, Any]
) -> list[dict[str, Any]]:
    scale = float(reference["scale_px_per_m"])
    rectangles = reference["wall_rectangles"]
    candidates: list[dict[str, Any]] = []
    for order, component in enumerate(elongated, start=1):
        cx, cy = component["centre_px"]
        centre_m = [cx / scale, -cy / scale]
        nearest = min(
            rectangles,
            key=lambda item: (
                (centre_m[0] - item["centre_m"][0]) ** 2
                + (centre_m[1] - item["centre_m"][1]) ** 2,
                item["dataset_rect_id"],
            ),
        )
        distance = float(
            np.linalg.norm(np.asarray(centre_m) - np.asarray(nearest["centre_m"]))
        )
        x, y, width, height = component["bbox_px"]
        candidates.append(
            {
                "candidate_id": f"U{order:03d}",
                **component,
                "centre_m_from_dataset_scale": [round(centre_m[0], 6), round(centre_m[1], 6)],
                "orientation_from_bbox": "vertical" if height >= width else "horizontal",
                "bbox_length_m": round(max(width, height) / scale, 6),
                "nearest_frozen_reference": nearest["label"],
                "nearest_reference_distance_m": round(distance, 6),
                "nearest_reference_continuous": bool(nearest["continuous"]),
                "claimed_as_element": False,
                "claim_failure": "no printed structural label and complete size",
            }
        )
    return candidates


def save_crops_and_manifest(
    image: np.ndarray, candidates: list[dict[str, Any]]
) -> dict[str, Any]:
    CROPS.mkdir(parents=True, exist_ok=True)
    for old in CROPS.glob("*.png"):
        old.unlink()
    flags: list[dict[str, Any]] = []
    height, width = image.shape[:2]
    for candidate in candidates:
        x, y, box_width, box_height = candidate["bbox_px"]
        margin = 96
        x0, y0 = max(0, x - margin), max(0, y - margin)
        x1, y1 = min(width, x + box_width + margin), min(height, y + box_height + margin)
        candidate_id = candidate["candidate_id"]
        relative = f"output/MLStructFP966/flags/crops/MLStructFP966-{candidate_id}-flag.png"
        destination = ROOT / relative
        write_image(destination, image[y0:y1, x0:x1])
        flags.append(
            {
                "flag_id": f"MLStructFP966:{candidate_id}:unclaimed_bar",
                "label": candidate_id,
                "flag": "unclaimed_bar_no_printed_label_or_size",
                "source": "locked classical shape-rule candidate",
                "question": (
                    f"For unclaimed candidate {candidate_id}, is one continuous structural "
                    "bar visible in this crop? No printed element label or complete size is "
                    f"available; use label {candidate_id} and size_mm NOT_VISIBLE, then answer "
                    "continuity only."
                ),
                "answer_schema": {
                    "label": candidate_id,
                    "size_mm": "NOT_VISIBLE",
                    "continuous": candidate["nearest_reference_continuous"],
                    "confidence": "0..1",
                    "reason": "short text",
                },
                "crop": relative,
                "crop_box_px": [x0, y0, x1, y1],
                "crop_sha256": sha256(destination),
                "nearest_frozen_reference": candidate["nearest_frozen_reference"],
            }
        )
    return {
        "plan": "MLStructFP966",
        "confidence_threshold": 0.85,
        "reference": "config/reference-dataset-MLStructFP966-HW3.json",
        "crops_frozen_before_vlm": True,
        "whole_plan_crop_prohibited": True,
        "question_order": [item["flag_id"] for item in flags],
        "flags": flags,
    }


def downsample(image: np.ndarray, max_width: int = 2400) -> np.ndarray:
    if image.shape[1] <= max_width:
        return image
    factor = max_width / image.shape[1]
    return cv2.resize(
        image,
        (max_width, int(round(image.shape[0] * factor))),
        interpolation=cv2.INTER_AREA,
    )


def plot_outputs(
    image: np.ndarray,
    masks: dict[str, np.ndarray],
    components: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    reference: dict[str, Any],
) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5), constrained_layout=True)
    display_masks = [
        ("white-composited source", cv2.cvtColor(downsample(image), cv2.COLOR_BGR2RGB), None),
        ("grey < 225", downsample(masks["ink"]), "gray"),
        ("11-px 45-degree opening", downsample(masks["diag"]), "gray"),
        ("17/21-px horizontal + vertical", downsample(cv2.bitwise_or(masks["horizontal"], masks["vertical"])), "gray"),
    ]
    for axis, (title, view, colour_map) in zip(axes.ravel(), display_masks):
        axis.imshow(view, cmap=colour_map)
        axis.set_title(title)
        axis.axis("off")
    fig.suptitle("MLStructFP966 - unchanged FloorPlan1 morphology", fontweight="bold")
    fig.savefig(FIGURES / "hw3_14_mlstruct_classical_morphology.png", dpi=180)
    plt.close(fig)

    overlay = image.copy()
    elongated_ids = {item["component_id"] for item in candidates}
    for component in components:
        x, y, width, height = component["bbox_px"]
        colour = (20, 20, 220) if component["component_id"] in elongated_ids else (30, 150, 230)
        cv2.rectangle(overlay, (x, y), (x + width, y + height), colour, 8)
    fig, axis = plt.subplots(figsize=(14, 9.5), constrained_layout=True)
    axis.imshow(cv2.cvtColor(downsample(overlay), cv2.COLOR_BGR2RGB))
    axis.set_title("20 components: red = 15 fixed 4:1 candidates, orange = rejected composites")
    axis.axis("off")
    fig.savefig(FIGURES / "hw3_15_mlstruct_connected_components.png", dpi=180)
    plt.close(fig)

    candidate_overlay = image.copy()
    reference_counts: dict[str, int] = {}
    for candidate in candidates:
        key = candidate["nearest_frozen_reference"]
        reference_counts[key] = reference_counts.get(key, 0) + 1
    duplicate_seen: set[str] = set()
    for candidate in candidates:
        x, y, width, height = candidate["bbox_px"]
        cv2.rectangle(candidate_overlay, (x, y), (x + width, y + height), (20, 20, 220), 8)
        reference_label = candidate["nearest_frozen_reference"]
        if reference_counts[reference_label] > 1:
            if reference_label in duplicate_seen:
                text_origin = (max(0, x - 210), min(candidate_overlay.shape[0] - 20, y + 95))
            else:
                text_origin = (min(candidate_overlay.shape[1] - 180, x + width + 25), y + 40)
                duplicate_seen.add(reference_label)
        else:
            text_origin = (x, max(30, y - 18))
        cv2.putText(
            candidate_overlay,
            candidate["candidate_id"],
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (20, 20, 220),
            4,
            cv2.LINE_AA,
        )
    fig, axis = plt.subplots(figsize=(14, 9.5), constrained_layout=True)
    axis.imshow(cv2.cvtColor(downsample(candidate_overlay), cv2.COLOR_BGR2RGB))
    axis.set_title("Label supervision stops: 15 unclaimed candidates, 0 printed C/T sizes")
    axis.axis("off")
    fig.savefig(FIGURES / "hw3_16_mlstruct_label_supervision_failure.png", dpi=180)
    plt.close(fig)

    comparison = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    scale = float(reference["scale_px_per_m"])
    for item in reference["wall_rectangles"]:
        x = int(round(item["centre_m"][0] * scale))
        y = int(round(-item["centre_m"][1] * scale))
        cv2.circle(comparison, (x, y), 20, (30, 130, 30), 7)
    for candidate in candidates:
        x, y = [int(round(value)) for value in candidate["centre_px"]]
        cv2.drawMarker(comparison, (x, y), (220, 40, 40), cv2.MARKER_CROSS, 45, 7)
    fig, axis = plt.subplots(figsize=(14, 9.5), constrained_layout=True)
    axis.imshow(downsample(comparison))
    axis.set_title("Audit only: red candidate centres versus 28 frozen dataset rectangles (green)")
    axis.axis("off")
    fig.savefig(FIGURES / "hw3_17_mlstruct_reference_comparison.png", dpi=180)
    plt.close(fig)


def main() -> None:
    params = read_json(PARAMS)
    plan = read_json(GENERALISATION)["plans"]["MLStructFP966"]
    reference = read_json(REFERENCE)
    cfg = params["classical"]
    source = ROOT / plan["source"]
    if sha256(source) != plan["sha256"]:
        raise RuntimeError("Frozen MLStructFP source hash changed")
    image = alpha_to_white(source)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    masks = morphology(gray, cfg)
    line_removed, line_mask = remove_long_lines(image, masks["ink"], cfg)

    engine = RapidOCR()
    ocr_original = serialise_ocr(engine(image), "original")
    ocr_no_lines = serialise_ocr(engine(line_removed), "long_lines_removed")
    accepted_ocr = [
        item
        for item in [*ocr_original, *ocr_no_lines]
        if item["confidence"] >= cfg["ocr_acceptance"]
    ]
    components, elongated = connected_component_sheet(
        masks["combined"], cfg["shape_ratio_min"]
    )
    candidates = candidate_sheet(elongated, reference)
    manifest = save_crops_and_manifest(image, candidates)

    unique_reference_hits = {
        item["nearest_frozen_reference"]
        for item in candidates
        if item["nearest_reference_distance_m"] <= 0.25
    }
    matched_candidates = sum(
        item["nearest_reference_distance_m"] <= 0.25 for item in candidates
    )
    record = {
        "source": {
            "file": source.name,
            "path": plan["source"],
            "sha256": plan["sha256"],
            "plan": "MLStructFP966",
            "preprocessing": plan["preprocessing"],
        },
        "elements": [],
        "unclaimed_candidates": candidates,
        "floor": {
            "pipeline_geometry": None,
            "reason": "the fixed label-supervision route cannot instantiate a floor polygon from this coloured-line source",
        },
        "method": {
            "locked_to_floorplan1_parameters": True,
            "parameter_file": "config/classical-params-HW3.json",
            "labels_supervise_one_element": True,
            "forest_selected": False,
            "vlm_used_for_coordinates": False,
            "reference_used_to_claim_elements": False,
        },
        "generalization_status": "failed safely at label supervision; no auditable element record promoted",
    }
    score = {
        "plan": "MLStructFP966",
        "image_shape_px": [int(image.shape[1]), int(image.shape[0])],
        "fixed_parameters": {
            "grey_threshold": cfg["grey_threshold"],
            "diagonal_kernel_px": cfg["diagonal_kernel_px"],
            "horizontal_open_px": cfg["horizontal_open_px"],
            "vertical_open_px": cfg["vertical_open_px"],
            "shape_ratio_min": cfg["shape_ratio_min"],
            "ocr_acceptance": cfg["ocr_acceptance"],
        },
        "ocr_detections_original": len(ocr_original),
        "ocr_detections_long_lines_removed": len(ocr_no_lines),
        "ocr_above_threshold": len(accepted_ocr),
        "printed_element_size_labels_reference": reference["printed_element_size_labels"],
        "connected_components_kept": len(components),
        "shape_rule_elongated_candidates": len(candidates),
        "label_claimed_elements": 0,
        "unclaimed_flags": len(manifest["flags"]),
        "audit_candidate_matches_within_0_25m": matched_candidates,
        "audit_unique_reference_rectangles_hit": len(unique_reference_hits),
        "audit_reference_rectangles_total": len(reference["wall_rectangles"]),
        "audit_unique_reference_recall": round(
            len(unique_reference_hits) / len(reference["wall_rectangles"]), 6
        ),
        "audit_duplicate_candidate_count": matched_candidates - len(unique_reference_hits),
        "structural_record_available": False,
    }

    write_json(RESULTS / "record-classical-label-supervised-MLStructFP966-HW3.json", record)
    write_json(RESULTS / "score-sheet-classical-MLStructFP966-HW3.json", score)
    write_json(RESULTS / "flags-manifest-MLStructFP966-HW3.json", manifest)
    write_json(INTERMEDIATE / "components-MLStructFP966-HW3.json", {"components": components, "elongated": elongated})
    write_json(INTERMEDIATE / "ocr-original-MLStructFP966-HW3.json", ocr_original)
    write_json(INTERMEDIATE / "ocr-long-lines-removed-MLStructFP966-HW3.json", ocr_no_lines)
    write_image(INTERMEDIATE / "mlstruct-composited-white.png", image)
    write_image(INTERMEDIATE / "mlstruct-threshold.png", masks["ink"])
    write_image(INTERMEDIATE / "mlstruct-diagonal-support.png", masks["diag"])
    write_image(INTERMEDIATE / "mlstruct-horizontal-open17.png", masks["horizontal"])
    write_image(INTERMEDIATE / "mlstruct-vertical-open21.png", masks["vertical"])
    write_image(INTERMEDIATE / "mlstruct-long-line-mask.png", line_mask)
    write_image(INTERMEDIATE / "mlstruct-long-lines-removed.png", line_removed)
    plot_outputs(image, masks, components, candidates, reference)
    print(json.dumps(score, indent=2))


if __name__ == "__main__":
    main()
