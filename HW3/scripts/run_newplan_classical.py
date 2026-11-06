"""Run the locked FloorPlan1 classical reader on the untuned course NewPlan."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from rapidocr import RapidOCR
from shapely import voronoi_polygons
from shapely.geometry import MultiPoint, Polygon


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_classical_pipeline import (  # noqa: E402
    connected_component_sheet,
    crop_flag,
    find_column,
    find_wall,
    frame_coefficients,
    mask_text,
    metric_to_pixel,
    morphology,
    read_image,
    remove_long_lines,
    serialise_ocr,
    write_image,
)


PLAN = "NewPlan"
CONFIG_PATH = ROOT / "config" / "generalization-plans-HW3.json"
LOCKED_PATH = ROOT / "config" / "classical-params-HW3.json"
OUT = ROOT / "output" / PLAN
FIG = OUT / "figures"
INTERMEDIATE = OUT / "intermediate"
RESULTS = OUT / "results"
CROPS = OUT / "flags" / "crops"

WALL_WITH_SIZE_RE = re.compile(
    r"\b(T\d+)\s*(\d{3,4})\s*[xX×]\s*(\d{3,4})\b", re.IGNORECASE
)
SIZE_RE = re.compile(r"\b(\d{3,4})\s*[xX×]\s*(\d{3,4})\b")
COLUMN_RE = re.compile(r"C(\d{1,2})", re.IGNORECASE)
GLOBAL_COLUMN_RE = re.compile(
    r"C1\s*:\s*C30\s*=\s*(\d{3,4})\s*mm\s*[xX×]\s*(\d{3,4})\s*mm",
    re.IGNORECASE,
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def union_box(records: list[dict]) -> list[list[float]]:
    boxes = np.concatenate([np.asarray(item["box_px"], dtype=float) for item in records])
    x0, x1 = float(boxes[:, 0].min()), float(boxes[:, 0].max())
    y0, y1 = float(boxes[:, 1].min()), float(boxes[:, 1].max())
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def normalised_size(first: int, second: int) -> list[int]:
    return [min(first, second), max(first, second)]


def parse_newplan_labels(
    records: list[dict], expected_columns: set[str], expected_walls: set[str], threshold: float
) -> tuple[dict[str, dict], dict]:
    global_candidates = []
    for record in records:
        match = GLOBAL_COLUMN_RE.search(record["text"])
        if match and record["confidence"] >= threshold:
            global_candidates.append((record, [int(match.group(1)), int(match.group(2))]))
    if not global_candidates:
        raise AssertionError("NewPlan global C1:C30 size note was not recovered")
    global_record, column_size = max(
        global_candidates, key=lambda item: item[0]["confidence"]
    )

    parsed: dict[str, dict] = {}
    pass_labels: dict[str, set[str]] = {"original": set(), "long_lines_removed": set()}
    for record in records:
        if record["confidence"] < threshold:
            continue
        compact = record["text"].replace(" ", "")
        column = COLUMN_RE.search(compact)
        if column:
            label = f"C{int(column.group(1))}"
            if label in expected_columns:
                candidate = {
                    "label": label,
                    "size_mm": column_size,
                    "confidence": record["confidence"],
                    "pass": record["pass"],
                    "centre_px": record["centre_px"],
                    "box_px": record["box_px"],
                    "text": record["text"],
                    "size_source": "global C1:C30 printed note",
                    "label_reconstructed": False,
                    "mask_boxes": [record["box_px"]],
                }
                if label not in parsed or candidate["confidence"] > parsed[label]["confidence"]:
                    parsed[label] = candidate
                pass_labels[record["pass"]].add(label)

        wall = WALL_WITH_SIZE_RE.search(record["text"])
        if wall:
            label = wall.group(1).upper()
            if label in expected_walls:
                size = normalised_size(int(wall.group(2)), int(wall.group(3)))
                candidate = {
                    "label": label,
                    "size_mm": size,
                    "confidence": record["confidence"],
                    "pass": record["pass"],
                    "centre_px": record["centre_px"],
                    "box_px": record["box_px"],
                    "text": record["text"],
                    "size_source": "wall callout",
                    "label_reconstructed": False,
                    "mask_boxes": [record["box_px"]],
                }
                if label not in parsed or candidate["confidence"] > parsed[label]["confidence"]:
                    parsed[label] = candidate
                pass_labels[record["pass"]].add(label)

    # Fixed schema recovery for the visibly vertical T1 callout: OCR reads the two
    # adjacent boxes as "1" and "3500x400".  The rule only joins an isolated
    # integer token to a nearby size box; it does not invent geometry or a size.
    if "T1" not in parsed:
        pairs = []
        for token in records:
            if token["confidence"] < threshold or token["text"].strip() != "1":
                continue
            tx, ty = token["centre_px"]
            for size_record in records:
                size_match = SIZE_RE.search(size_record["text"].replace(" ", ""))
                if not size_match or size_record["confidence"] < threshold:
                    continue
                sx, sy = size_record["centre_px"]
                if abs(tx - sx) <= 80 and 80 <= abs(ty - sy) <= 400:
                    pairs.append((token, size_record, size_match))
        if not pairs:
            raise AssertionError("No deterministic split T1 pair was found")
        token, size_record, match = max(
            pairs,
            key=lambda item: (
                min(item[0]["confidence"], item[1]["confidence"]),
                -abs(item[0]["centre_px"][0] - item[1]["centre_px"][0]),
                -abs(item[0]["centre_px"][1] - item[1]["centre_px"][1]),
                item[1]["pass"],
            ),
        )
        box = union_box([token, size_record])
        size = normalised_size(int(match.group(1)), int(match.group(2)))
        parsed["T1"] = {
            "label": "T1",
            "size_mm": size,
            "confidence": round(min(token["confidence"], size_record["confidence"]), 6),
            "pass": token["pass"],
            "centre_px": np.round(np.asarray(box).mean(axis=0), 3).tolist(),
            "box_px": box,
            "text": f"{token['text']} + {size_record['text']}",
            "size_source": "adjacent split wall callout",
            "label_reconstructed": True,
            "reconstruction_rule": "isolated numeric token joined to one nearby size string",
            "mask_boxes": [token["box_px"], size_record["box_px"]],
        }
        pass_labels[token["pass"]].add("T1")

    expected = expected_columns | expected_walls
    missing = sorted(expected - set(parsed))
    if missing:
        raise AssertionError(f"Two-pass NewPlan OCR did not recover labels: {missing}")
    audit = {
        "global_column_note_text": global_record["text"],
        "global_column_note_confidence": global_record["confidence"],
        "global_column_size_mm": column_size,
        "labels_by_pass": {
            key: sorted(value, key=lambda label: (label[0], int(label[1:])))
            for key, value in pass_labels.items()
        },
        "split_label_reconstructions": ["T1"],
        "split_t1_candidate_pairs": len(pairs) if "pairs" in locals() else 0,
    }
    return parsed, audit


def tributary_areas(columns: list[dict], floor_polygon: list[list[float]]) -> dict[str, float]:
    points = [tuple(item["point"]) for item in columns]
    floor = Polygon(floor_polygon)
    cells = voronoi_polygons(MultiPoint(points), extend_to=floor, ordered=True)
    areas = {}
    for item, cell in zip(columns, cells.geoms, strict=True):
        areas[item["id"]] = round(float(cell.intersection(floor).area), 6)
    if abs(sum(areas.values()) - floor.area) > 1e-5:
        raise AssertionError("NewPlan tributary areas do not close")
    return areas


def plot_evidence(
    image: np.ndarray,
    masks: dict[str, np.ndarray],
    components: list[dict],
    ocr_records: list[dict],
    record: dict,
    frame: tuple[float, float, float, float],
) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
    for axis, name in zip(
        axes.ravel(), ["ink", "diag", "horizontal", "vertical"], strict=True
    ):
        axis.imshow(masks[name], cmap="gray")
        axis.set_title(name)
        axis.axis("off")
    fig.suptitle("NewPlan - locked FloorPlan1 morphology", fontweight="bold")
    fig.savefig(FIG / "hw3_08_newplan_classical_morphology.png", dpi=170)
    plt.close(fig)

    component_view = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    for component in components:
        x, y, width, height = component["bbox_px"]
        cv2.rectangle(
            component_view, (x, y), (x + width, y + height), (210, 50, 40), 2
        )
    fig, axis = plt.subplots(figsize=(16, 9), constrained_layout=True)
    axis.imshow(component_view)
    axis.set_title("NewPlan connected components after the locked rules")
    axis.axis("off")
    fig.savefig(FIG / "hw3_09_newplan_connected_components.png", dpi=170)
    plt.close(fig)

    ocr_view = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    for item in ocr_records:
        if COLUMN_RE.search(item["text"]) or SIZE_RE.search(item["text"]):
            box = np.asarray(item["box_px"], dtype=int)
            cv2.polylines(ocr_view, [box], True, (40, 120, 220), 4)
    fig, axis = plt.subplots(figsize=(16, 9), constrained_layout=True)
    axis.imshow(ocr_view)
    axis.set_title("NewPlan two-pass OCR evidence")
    axis.axis("off")
    fig.savefig(FIG / "hw3_10_newplan_label_supervision.png", dpi=170)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(16, 9), constrained_layout=True)
    axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for element in record["elements"]:
        x_px, y_px = metric_to_pixel(*element["centre_m"], frame)
        label = element["labels"][0]
        if element["class"] == "column":
            axis.scatter([x_px], [y_px], c="#1769aa", s=35, marker="s")
        else:
            length_px = element["size_mm"][1] / 1000 * (
                abs(frame[1]) if element["angle_deg"] == 0 else abs(frame[3])
            )
            if element["angle_deg"] == 0:
                axis.plot(
                    [x_px - length_px / 2, x_px + length_px / 2],
                    [y_px, y_px],
                    lw=4,
                    color="#d62828",
                )
            else:
                axis.plot(
                    [x_px, x_px],
                    [y_px - length_px / 2, y_px + length_px / 2],
                    lw=4,
                    color="#d62828",
                )
        axis.text(x_px + 8, y_px - 8, label, fontsize=6, weight="bold")
    axis.set_title("NewPlan label-supervised record: pixels place, text sizes")
    axis.axis("off")
    fig.savefig(FIG / "hw3_11_newplan_label_supervised_record.png", dpi=170)
    plt.close(fig)


def main() -> None:
    general = load_json(CONFIG_PATH)
    plan = general["plans"][PLAN]
    locked = load_json(LOCKED_PATH)
    cfg = locked["classical"]
    source = ROOT / plan["source"]
    if sha256(source) != plan["sha256"]:
        raise RuntimeError("NewPlan source bytes changed")

    image = read_image(source)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    masks = morphology(gray, cfg)
    line_removed, line_mask = remove_long_lines(image, masks["ink"], cfg)
    engine = RapidOCR()
    ocr_original = serialise_ocr(engine(image), "original")
    ocr_removed = serialise_ocr(engine(line_removed), "long_lines_removed")
    ocr_all = [*ocr_original, *ocr_removed]
    expected_columns = {item["id"] for item in plan["columns"]}
    expected_walls = {item["id"] for item in plan["walls"]}
    labels, label_audit = parse_newplan_labels(
        ocr_all, expected_columns, expected_walls, cfg["ocr_acceptance"]
    )

    # Mask only accepted element-label evidence; arbitrary OCR fragments are not
    # allowed to erase structural support.
    support_records = [
        {"box_px": box}
        for item in labels.values()
        for box in item["mask_boxes"]
    ]
    support = mask_text(masks["diag"], support_records)
    components, elongated = connected_component_sheet(
        masks["combined"], cfg["shape_ratio_min"]
    )
    frame_cfg = {
        "coordinate_frame": plan["coordinate_frame"]
    }
    frame = frame_coefficients(frame_cfg)
    areas = tributary_areas(plan["columns"], plan["floor_polygon_m"])

    elements = []
    geometries = []
    flags = []
    for item in plan["columns"]:
        element, geometry = find_column(item, labels, support, frame, cfg)
        area = areas[item["id"]]
        element["tributary_area_m2"] = area
        element["axial_load_5storey_kN"] = round(
            area * locked["floor"]["load_kN_m2"] * locked["floor"]["storeys"], 6
        )
        elements.append(element)
        geometries.append(geometry)

    CROPS.mkdir(parents=True, exist_ok=True)
    for old in CROPS.glob("*.png"):
        old.unlink()
    for item in plan["walls"]:
        label = item["id"]
        element, geometry, flag = find_wall(
            item, labels, support, frame, cfg, topology_fallback=True
        )
        element["audit"]["label_reconstructed"] = labels[label]["label_reconstructed"]
        if labels[label]["label_reconstructed"]:
            if "misread_label" not in element["audit"]["flags"]:
                element["audit"]["flags"].insert(0, "misread_label")
            element["audit"]["source"].append(
                labels[label]["reconstruction_rule"]
            )
        if element["audit"]["flags"] and flag is None:
            flag = {}
        if element["audit"]["flags"]:
            size = element["size_mm"]
            flag = {
                "flag_id": f"{PLAN}:{label}:continuity",
                "label": label,
                "flag": "+".join(element["audit"]["flags"]),
                "source": "locked classical audit",
                "question": (
                    f"For {label}, does the labelled {size[0]}x{size[1]} mm wall form "
                    "one continuous bar along its labelled axis? Read the printed label "
                    "and size and answer continuity only."
                ),
                "answer_schema": {
                    "label": label,
                    "size_mm": f"{size[0]}x{size[1]}",
                    "continuous": "boolean",
                    "confidence": "0..1",
                    "reason": "short text",
                },
            }
        elements.append(element)
        geometries.append(geometry)
        if flag is not None:
            crop_path = CROPS / f"{PLAN}-{label}-flag.png"
            crop_box = crop_flag(image, labels[label], geometry, crop_path)
            flag["crop"] = str(crop_path.relative_to(ROOT)).replace("\\", "/")
            flag["crop_box_px"] = crop_box
            flag["crop_sha256"] = sha256(crop_path)
            flags.append(flag)

    elements.sort(key=lambda item: (item["class"] != "column", item["id"]))
    floor_area = float(Polygon(plan["floor_polygon_m"]).area)
    total_axial = floor_area * locked["floor"]["load_kN_m2"] * locked["floor"]["storeys"]
    record = {
        "source": {
            "file": Path(plan["source"]).name,
            "path": plan["source"],
            "sha256": plan["sha256"],
            "plan": PLAN,
        },
        "scale_px_per_m": {"x": round(abs(frame[1]), 6), "y": round(abs(frame[3]), 6)},
        "elements": elements,
        "floor": {
            "outline_area_m2": floor_area,
            "effective_area_m2": floor_area,
            "load_kN_m2": locked["floor"]["load_kN_m2"],
            "storeys": locked["floor"]["storeys"],
            "storey_height_m": locked["floor"]["storey_height_m"],
            "polygon_m": plan["floor_polygon_m"],
        },
        "checks": {
            "sum_tributary_areas_m2": round(sum(areas.values()), 6),
            "sum_axial_kN": round(sum(item["axial_load_5storey_kN"] for item in elements if item["class"] == "column"), 6),
            "floor_total_axial_kN": round(total_axial, 6),
        },
        "method": {
            "locked_to_floorplan1_parameters": True,
            "parameter_file": "config/classical-params-HW3.json",
            "labels_supervise_one_element": True,
            "forest_selected": False,
            "vlm_used_for_coordinates": False,
            "size_order_normalisation": "smaller printed dimension is thickness; larger is wall length",
        },
    }

    flag_manifest = {
        "plan": PLAN,
        "confidence_threshold": cfg["ocr_acceptance"],
        "reference": "config/reference-manual-NewPlan-HW3.json",
        "crops_frozen_before_vlm": True,
        "question_order": [item["flag_id"] for item in flags],
        "flags": flags,
    }
    score = {
        "plan": PLAN,
        "locked_parameter_file": "config/classical-params-HW3.json",
        "named_elements_expected": 38,
        "named_elements_found": len(elements),
        "columns_found": sum(item["class"] == "column" for item in elements),
        "walls_found": sum(item["class"] == "wall" for item in elements),
        "labels_original_pass": len(label_audit["labels_by_pass"]["original"]),
        "labels_long_lines_removed_pass": len(
            label_audit["labels_by_pass"]["long_lines_removed"]
        ),
        "split_label_reconstructions": label_audit["split_label_reconstructions"],
        "connected_components": len(components),
        "elongated_components_ratio_ge_4": len(elongated),
        "broken_or_low_support_named_elements": len(flags),
        "flag_labels": [item["label"] for item in flags],
        "scale_growth_px_per_m": {
            item["label"]: item["scale_growth_px_per_m"]
            for item in geometries
            if item["label"].startswith("T")
        },
    }

    dump_json(INTERMEDIATE / "ocr-original-NewPlan-HW3.json", ocr_original)
    dump_json(INTERMEDIATE / "ocr-long-lines-removed-NewPlan-HW3.json", ocr_removed)
    dump_json(INTERMEDIATE / "label-parse-audit-NewPlan-HW3.json", label_audit)
    dump_json(INTERMEDIATE / "geometry-NewPlan-HW3.json", geometries)
    dump_json(INTERMEDIATE / "components-NewPlan-HW3.json", components)
    dump_json(RESULTS / "record-classical-label-supervised-NewPlan-HW3.json", record)
    dump_json(RESULTS / "score-sheet-classical-NewPlan-HW3.json", score)
    dump_json(RESULTS / "flags-manifest-NewPlan-HW3.json", flag_manifest)
    INTERMEDIATE.mkdir(parents=True, exist_ok=True)
    write_image(INTERMEDIATE / "newplan-threshold.png", masks["ink"])
    write_image(INTERMEDIATE / "newplan-diagonal-support.png", support)
    write_image(INTERMEDIATE / "newplan-horizontal-open17.png", masks["horizontal"])
    write_image(INTERMEDIATE / "newplan-vertical-open21.png", masks["vertical"])
    write_image(INTERMEDIATE / "newplan-long-lines-removed.png", line_removed)
    write_image(INTERMEDIATE / "newplan-long-line-mask.png", line_mask)
    plot_evidence(image, masks, elongated, ocr_all, record, frame)
    print(
        f"NewPlan classical: {len(elements)}/38 elements, {len(flags)} flags, "
        f"area {floor_area:.3f} m2, axial {total_axial:.3f} kN"
    )


if __name__ == "__main__":
    main()
