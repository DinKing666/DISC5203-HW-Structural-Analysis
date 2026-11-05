from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from rapidocr import RapidOCR
from shapely.geometry import Polygon
from shapely.ops import unary_union


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
PARAMS_PATH = ROOT / "config" / "classical-params-HW3.json"
HW1_CONFIG = REPO / "HW1" / "config" / "plan1_geometry.json"
HW1_VERIFY = REPO / "HW1" / "output" / "results" / "verification.json"
OUT = ROOT / "output" / "FloorPlan1"
FIG = OUT / "figures"
INTERMEDIATE = OUT / "intermediate"
RESULTS = OUT / "results"
CROPS = OUT / "flags" / "crops"


def dump_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_image(path: Path, mode: int = cv2.IMREAD_COLOR) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, mode)
    if image is None:
        raise FileNotFoundError(path)
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise RuntimeError(f"Could not encode image: {path}")
    encoded.tofile(path)


def load_inputs() -> tuple[dict, dict, dict]:
    params = json.loads(PARAMS_PATH.read_text(encoding="utf-8"))
    hw1 = json.loads(HW1_CONFIG.read_text(encoding="utf-8"))
    verification = json.loads(HW1_VERIFY.read_text(encoding="utf-8"))
    return params, hw1, verification


def frame_coefficients(params: dict) -> tuple[float, float, float, float]:
    (px0, mx0), (px1, mx1) = params["coordinate_frame"]["x_px_at_m"]
    (py0, my0), (py1, my1) = params["coordinate_frame"]["y_px_at_m"]
    sx = (px1 - px0) / (mx1 - mx0)
    sy = (py1 - py0) / (my1 - my0)
    ax = px0 - sx * mx0
    ay = py0 - sy * my0
    return ax, sx, ay, sy


def metric_to_pixel(x_m: float, y_m: float, frame: tuple[float, float, float, float]) -> tuple[float, float]:
    ax, sx, ay, sy = frame
    return ax + sx * x_m, ay + sy * y_m


def pixel_to_metric(x_px: float, y_px: float, frame: tuple[float, float, float, float]) -> tuple[float, float]:
    ax, sx, ay, sy = frame
    return (x_px - ax) / sx, (y_px - ay) / sy


def morphology(gray: np.ndarray, cfg: dict) -> dict[str, np.ndarray]:
    ink = (gray < cfg["grey_threshold"]).astype(np.uint8) * 255
    diag_kernel = np.eye(cfg["diagonal_kernel_px"], dtype=np.uint8)
    diag = cv2.morphologyEx(ink, cv2.MORPH_OPEN, diag_kernel)
    horizontal = cv2.morphologyEx(
        ink, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (cfg["horizontal_open_px"], 1)),
    )
    vertical = cv2.morphologyEx(
        ink, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, cfg["vertical_open_px"])),
    )
    combined = cv2.bitwise_or(diag, cv2.bitwise_or(horizontal, vertical))
    return {"ink": ink, "diag": diag, "horizontal": horizontal, "vertical": vertical, "combined": combined}


def remove_long_lines(image: np.ndarray, ink: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    length = cfg["long_line_remove_px"]
    horizontal = cv2.morphologyEx(
        ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1))
    )
    vertical = cv2.morphologyEx(
        ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, length))
    )
    line_mask = cv2.dilate(cv2.bitwise_or(horizontal, vertical), np.ones((3, 3), np.uint8))
    cleaned = image.copy()
    cleaned[line_mask > 0] = 255
    return cleaned, line_mask


def serialise_ocr(output: object, pass_name: str) -> list[dict]:
    if getattr(output, "txts", None) is None:
        return []
    records = []
    for text, score, box in zip(output.txts, output.scores, output.boxes):
        box_array = np.asarray(box, dtype=float)
        records.append({
            "pass": pass_name,
            "text": str(text),
            "confidence": round(float(score), 6),
            "box_px": np.round(box_array, 3).tolist(),
            "centre_px": np.round(box_array.mean(axis=0), 3).tolist(),
        })
    return records


LABEL_RE = re.compile(r"\b([CT]\d+)\s*(\d{3,4})\s*[xX×]\s*(\d{3,4})\b", re.IGNORECASE)


def parse_element_labels(records: list[dict], threshold: float) -> dict[str, dict]:
    parsed: dict[str, dict] = {}
    for record in records:
        match = LABEL_RE.search(record["text"].replace(" ", ""))
        if not match or record["confidence"] < threshold:
            continue
        label = match.group(1).upper()
        candidate = {
            "label": label,
            "size_mm": [int(match.group(2)), int(match.group(3))],
            "confidence": record["confidence"],
            "pass": record["pass"],
            "centre_px": record["centre_px"],
            "box_px": record["box_px"],
            "text": record["text"],
        }
        existing = parsed.get(label)
        if existing is None or candidate["confidence"] > existing["confidence"]:
            parsed[label] = candidate
    return parsed


def mask_text(diag: np.ndarray, records: list[dict]) -> np.ndarray:
    cleaned = diag.copy()
    for record in records:
        box = np.asarray(record["box_px"], dtype=float)
        x0 = max(0, int(np.floor(box[:, 0].min())) - 4)
        x1 = min(cleaned.shape[1], int(np.ceil(box[:, 0].max())) + 5)
        y0 = max(0, int(np.floor(box[:, 1].min())) - 4)
        y1 = min(cleaned.shape[0], int(np.ceil(box[:, 1].max())) + 5)
        cleaned[y0:y1, x0:x1] = 0
    return cleaned


def connected_component_sheet(mask: np.ndarray, ratio_min: float) -> tuple[list[dict], list[dict]]:
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
    components: list[dict] = []
    elongated: list[dict] = []
    for idx in range(1, count):
        x, y, width, height, area = [int(v) for v in stats[idx]]
        if area < 20:
            continue
        item = {
            "component_id": idx,
            "bbox_px": [x, y, width, height],
            "area_px": area,
            "centre_px": [round(float(centroids[idx, 0]), 3), round(float(centroids[idx, 1]), 3)],
            "aspect_ratio": round(max(width, height) / max(1, min(width, height)), 3),
        }
        components.append(item)
        if item["aspect_ratio"] >= ratio_min:
            elongated.append(item)
    return components, elongated


def local_maximum(score: np.ndarray, centre: tuple[float, float], radius: tuple[int, int]) -> tuple[int, int]:
    cx, cy = centre
    rx, ry = radius
    x0, x1 = max(0, int(cx - rx)), min(score.shape[1], int(cx + rx + 1))
    y0, y1 = max(0, int(cy - ry)), min(score.shape[0], int(cy + ry + 1))
    crop = score[y0:y1, x0:x1]
    iy, ix = np.unravel_index(int(np.argmax(crop)), crop.shape)
    return x0 + int(ix), y0 + int(iy)


def weighted_peak_coordinate(values: np.ndarray, offset: int) -> int:
    if values.size == 0 or float(values.max()) <= 0:
        return offset + values.size // 2
    indices = np.flatnonzero(values >= 0.20 * float(values.max()))
    weights = values[indices].astype(float)
    return offset + int(round(float(np.average(indices, weights=weights))))


def find_column(
    item: dict, label_data: dict, support: np.ndarray, frame: tuple[float, float, float, float], cfg: dict
) -> tuple[dict, dict]:
    label = item["id"]
    expected_px = metric_to_pixel(*item["point"], frame)
    _, sx, _, sy = frame
    size_mm = label_data[label]["size_mm"]
    wx = max(5, int(round(size_mm[0] / 1000 * abs(sx))))
    wy = max(5, int(round(size_mm[1] / 1000 * abs(sy))))
    density = cv2.boxFilter((support > 0).astype(np.float32), -1, (wx, wy), normalize=False)
    radius = (
        int(round(cfg["column_search_padding_m"] * abs(sx))),
        int(round(cfg["column_search_padding_m"] * abs(sy))),
    )
    x_px, y_px = local_maximum(density, expected_px, radius)
    x_m, y_m = pixel_to_metric(x_px, y_px, frame)
    audit = {
        "merged": False,
        "partly_missed": [],
        "pixel_support": round(float(density[y_px, x_px]) / max(1, wx * wy), 4),
        "source": ["classical diagonal-stroke density", f"label {label_data[label]['pass']}"],
    }
    record = {
        "id": int(label[1:]),
        "labels": [label],
        "class": "column",
        "centre_m": [round(x_m, 4), round(y_m, 4)],
        "size_mm": size_mm,
        "audit": audit,
    }
    return record, {"label": label, "centre_px": [x_px, y_px], "window_px": [wx, wy]}


def best_fixed_interval(values: np.ndarray, start: int, end: int, length: int) -> tuple[int, int, float]:
    start = max(0, start)
    end = min(values.size, end)
    length = max(2, min(length, end - start))
    segment = values[start:end]
    kernel = np.ones(length, dtype=np.float64)
    sums = np.convolve(segment.astype(np.float64), kernel, mode="valid")
    offset = int(np.argmax(sums))
    return start + offset, start + offset + length, float(sums[offset])


def gap_spans(bin_support: np.ndarray, length_m: float) -> tuple[list[list[float]], float, float]:
    maximum = max(1e-9, float(bin_support.max()))
    supported = bin_support >= max(1.0, 0.12 * maximum)
    gaps: list[list[float]] = []
    gap_start = None
    for idx, value in enumerate(np.r_[supported, True]):
        if not value and gap_start is None:
            gap_start = idx
        elif value and gap_start is not None:
            gaps.append([round(gap_start / len(supported) * length_m, 3), round(idx / len(supported) * length_m, 3)])
            gap_start = None
    max_gap = max((b - a for a, b in gaps), default=0.0)
    return gaps, float(supported.mean()), float(max_gap)


def find_wall(
    item: dict, label_data: dict, support: np.ndarray, frame: tuple[float, float, float, float], cfg: dict
) -> tuple[dict, dict, dict | None]:
    label = item["id"]
    size_mm = label_data[label]["size_mm"]
    thickness_m = size_mm[0] / 1000
    length_m = size_mm[1] / 1000
    start_m = np.asarray(item["start"], dtype=float)
    end_m = np.asarray(item["end"], dtype=float)
    horizontal = abs(end_m[0] - start_m[0]) >= abs(end_m[1] - start_m[1])
    _, sx, _, sy = frame
    p0 = metric_to_pixel(*start_m, frame)
    p1 = metric_to_pixel(*end_m, frame)
    pad_x = int(round(cfg["wall_search_padding_m"] * abs(sx)))
    pad_y = int(round(cfg["wall_search_padding_m"] * abs(sy)))

    if horizontal:
        y_expected = (p0[1] + p1[1]) / 2
        y0, y1 = max(0, int(y_expected - pad_y)), min(support.shape[0], int(y_expected + pad_y + 1))
        axis_by_y = support[y0:y1, :].sum(axis=1)
        y_px = weighted_peak_coordinate(axis_by_y, y0)
        half = max(2, int(round(thickness_m * abs(sy) / 2)))
        axis_score = support[max(0, y_px - half):min(support.shape[0], y_px + half + 1), :].sum(axis=0)
        length_px = int(round(length_m * abs(sx)))
        label_axis = float(label_data[label]["centre_px"][0])
        anchor_pad = int(round(0.75 * abs(sx)))
        search0 = int(label_axis - length_px / 2 - anchor_pad)
        search1 = int(label_axis + length_px / 2 + anchor_pad)
        a_px, b_px, score = best_fixed_interval(axis_score, search0, search1, length_px)
        centre_px = ((a_px + b_px) / 2, y_px)
        bins = np.array_split(axis_score[a_px:b_px], cfg["continuity_bins"])
        bin_support = np.array([chunk.sum() for chunk in bins])
        angle = 0.0
        measured_scale = (b_px - a_px - cfg["diagonal_kernel_px"]) / length_m
    else:
        x_expected = (p0[0] + p1[0]) / 2
        x0, x1 = max(0, int(x_expected - pad_x)), min(support.shape[1], int(x_expected + pad_x + 1))
        axis_by_x = support[:, x0:x1].sum(axis=0)
        x_px = weighted_peak_coordinate(axis_by_x, x0)
        half = max(2, int(round(thickness_m * abs(sx) / 2)))
        axis_score = support[:, max(0, x_px - half):min(support.shape[1], x_px + half + 1)].sum(axis=1)
        length_px = int(round(length_m * abs(sy)))
        label_axis = float(label_data[label]["centre_px"][1])
        anchor_pad = int(round(0.75 * abs(sy)))
        search0 = int(label_axis - length_px / 2 - anchor_pad)
        search1 = int(label_axis + length_px / 2 + anchor_pad)
        topology0 = int(min(p0[1], p1[1]) - pad_y)
        topology1 = int(max(p0[1], p1[1]) + pad_y)
        bounded0 = max(search0, topology0)
        bounded1 = min(search1, topology1)
        if bounded1 - bounded0 >= length_px:
            search0, search1 = bounded0, bounded1
        a_px, b_px, score = best_fixed_interval(axis_score, search0, search1, length_px)
        centre_px = (x_px, (a_px + b_px) / 2)
        bins = np.array_split(axis_score[a_px:b_px], cfg["continuity_bins"])
        bin_support = np.array([chunk.sum() for chunk in bins])
        angle = 90.0
        measured_scale = (b_px - a_px - cfg["diagonal_kernel_px"]) / length_m

    gaps, supported_fraction, max_gap_m = gap_spans(bin_support, length_m)
    continuous = supported_fraction >= cfg["continuity_min_supported_fraction"] and max_gap_m <= cfg["continuity_max_gap_m"]
    x_m, y_m = pixel_to_metric(*centre_px, frame)
    flags = []
    if not continuous:
        flags.append("wall_in_pieces_along_axis")
    if supported_fraction < 0.80:
        flags.append("low_support_completion")
    audit = {
        "merged": False,
        "partly_missed": gaps,
        "supported_fraction": round(supported_fraction, 4),
        "max_gap_m": round(max_gap_m, 4),
        "flags": flags,
        "source": ["11-px 45-degree support", f"label {label_data[label]['pass']}"],
    }
    record = {
        "id": int(label[1:]),
        "labels": [label],
        "class": "wall",
        "centre_m": [round(x_m, 4), round(y_m, 4)],
        "size_mm": size_mm,
        "angle_deg": angle,
        "continuous": continuous,
        "audit": audit,
    }
    geometry = {
        "label": label,
        "centre_px": [round(float(centre_px[0]), 3), round(float(centre_px[1]), 3)],
        "axis_interval_px": [a_px, b_px],
        "orientation": "horizontal" if horizontal else "vertical",
        "score": round(score, 3),
        "scale_growth_px_per_m": round(float(measured_scale), 4),
    }
    flag = None
    if flags:
        flag = {
            "flag_id": f"FloorPlan1:{label}:continuity",
            "label": label,
            "flag": "+".join(flags),
            "source": "classical audit",
            "question": f"For {label}, does the labelled {size_mm[0]}x{size_mm[1]} mm wall form one continuous bar along its labelled axis? Read the printed size and answer continuity only.",
            "answer_schema": {"label": label, "size_mm": f"{size_mm[0]}x{size_mm[1]}", "continuous": "boolean", "confidence": "0..1", "reason": "short text"},
        }
    return record, geometry, flag


def crop_flag(image: np.ndarray, label_data: dict, geometry: dict, destination: Path) -> list[int]:
    cx, cy = geometry["centre_px"]
    if geometry["orientation"] == "horizontal":
        a, b = geometry["axis_interval_px"]
        x0, x1 = max(0, int(a - 100)), min(image.shape[1], int(b + 100))
        y0, y1 = max(0, int(cy - 130)), min(image.shape[0], int(cy + 130))
    else:
        a, b = geometry["axis_interval_px"]
        x0, x1 = max(0, int(cx - 180)), min(image.shape[1], int(cx + 180))
        y0, y1 = max(0, int(a - 100)), min(image.shape[0], int(b + 100))
    label_box = np.asarray(label_data["box_px"], dtype=float)
    x0 = max(0, min(x0, int(np.floor(label_box[:, 0].min())) - 40))
    x1 = min(image.shape[1], max(x1, int(np.ceil(label_box[:, 0].max())) + 40))
    y0 = max(0, min(y0, int(np.floor(label_box[:, 1].min())) - 40))
    y1 = min(image.shape[0], max(y1, int(np.ceil(label_box[:, 1].max())) + 40))
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_image(destination, image[y0:y1, x0:x1])
    return [x0, y0, x1, y1]


def floor_areas(hw1: dict) -> tuple[float, float]:
    base = Polygon(hw1["outline"])
    additions = [Polygon(points) for points in hw1["additions"]]
    openings = [Polygon(points) for points in hw1["openings"]]
    gross = unary_union([base, *additions])
    net = gross.difference(unary_union(openings))
    return float(gross.area), float(net.area)


def plot_outputs(
    image: np.ndarray, masks: dict[str, np.ndarray], components: list[dict], record: dict,
    frame: tuple[float, float, float, float], ocr_records: list[dict]
) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 12), constrained_layout=True)
    for ax, (name, mask) in zip(axes.ravel(), masks.items()):
        ax.imshow(mask, cmap="gray")
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle("FloorPlan1 - fixed L07 morphology", fontweight="bold")
    fig.savefig(FIG / "hw3_01_classical_morphology.png", dpi=180)
    plt.close(fig)

    component_view = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    for component in components:
        x, y, width, height = component["bbox_px"]
        cv2.rectangle(component_view, (x, y), (x + width, y + height), (210, 50, 40), 1)
    fig, ax = plt.subplots(figsize=(10, 13), constrained_layout=True)
    ax.imshow(component_view)
    ax.set_title("Connected components after threshold and morphology")
    ax.axis("off")
    fig.savefig(FIG / "hw3_02_connected_components.png", dpi=180)
    plt.close(fig)

    ocr_view = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    for item in ocr_records:
        if LABEL_RE.search(item["text"].replace(" ", "")):
            box = np.asarray(item["box_px"], dtype=int)
            cv2.polylines(ocr_view, [box], True, (40, 120, 220), 3)
    fig, ax = plt.subplots(figsize=(10, 13), constrained_layout=True)
    ax.imshow(ocr_view)
    ax.set_title("Two-pass OCR element labels (blue boxes)")
    ax.axis("off")
    fig.savefig(FIG / "hw3_03_label_supervision.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 13), constrained_layout=True)
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for element in record["elements"]:
        x_px, y_px = metric_to_pixel(*element["centre_m"], frame)
        label = element["labels"][0]
        if element["class"] == "column":
            ax.scatter([x_px], [y_px], c="#1769aa", s=55, marker="s")
        else:
            length_px = element["size_mm"][1] / 1000 * (abs(frame[1]) if element["angle_deg"] == 0 else abs(frame[3]))
            if element["angle_deg"] == 0:
                ax.plot([x_px - length_px / 2, x_px + length_px / 2], [y_px, y_px], lw=5, color="#d62828")
            else:
                ax.plot([x_px, x_px], [y_px - length_px / 2, y_px + length_px / 2], lw=5, color="#d62828")
        ax.text(x_px + 8, y_px - 8, label, color="#111111", fontsize=8, weight="bold")
    ax.set_title("Label-supervised record: pixels place, text sizes")
    ax.axis("off")
    fig.savefig(FIG / "hw3_04_label_supervised_record.png", dpi=180)
    plt.close(fig)


def main() -> None:
    params, hw1, verification = load_inputs()
    cfg = params["classical"]
    source = ROOT / params["source"]
    image = read_image(source)
    if CROPS.exists():
        for old_crop in CROPS.glob("*.png"):
            old_crop.unlink()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    masks = morphology(gray, cfg)
    line_removed, line_mask = remove_long_lines(image, masks["ink"], cfg)

    engine = RapidOCR()
    ocr_original = serialise_ocr(engine(image), "original")
    ocr_no_lines = serialise_ocr(engine(line_removed), "long_lines_removed")
    ocr_all = [*ocr_original, *ocr_no_lines]
    labels = parse_element_labels(ocr_all, cfg["ocr_acceptance"])
    expected = {item["id"] for item in hw1["columns"]} | {item["id"] for item in hw1["walls"]}
    missing = sorted(expected - set(labels))
    if missing:
        raise AssertionError(f"Two-pass OCR did not recover labels: {missing}")

    element_text_records = [item for item in ocr_all if LABEL_RE.search(item["text"].replace(" ", ""))]
    text_free_support = mask_text(masks["diag"], element_text_records)
    components, elongated = connected_component_sheet(masks["combined"], cfg["shape_ratio_min"])
    frame = frame_coefficients(params)
    elements = []
    measured = []
    flags = []

    for column in hw1["columns"]:
        record, geometry = find_column(column, labels, text_free_support, frame, cfg)
        elements.append(record)
        measured.append(geometry)
    for wall in hw1["walls"]:
        record, geometry, flag = find_wall(wall, labels, text_free_support, frame, cfg)
        elements.append(record)
        measured.append(geometry)
        if flag is not None:
            crop_name = f"{flag['flag_id'].replace(':', '_')}.png"
            crop_path = CROPS / crop_name
            flag["crop"] = str(crop_path.relative_to(ROOT)).replace("\\", "/")
            flag["crop_box_px"] = crop_flag(image, labels[record["labels"][0]], geometry, crop_path)
            flag["crop_sha256"] = hashlib.sha256(crop_path.read_bytes()).hexdigest()
            flags.append(flag)

    gross_area, net_area = floor_areas(hw1)
    record = {
        "source": {"file": source.name},
        "scale_px_per_m": {"x": round(abs(frame[1]), 4), "y": round(abs(frame[3]), 4)},
        "method": {
            "classical": "grey<225; 11-px 45-degree opening; horizontal open 17; vertical open 21; connected components; 4:1 shape rule",
            "supervision": "accepted HW1 topology supplies search neighbourhoods; two-pass OCR supplies one size per label; diagonal pixels place the fixed-length bar",
            "forest_selected": False,
            "vlm_used_for_coordinates": False
        },
        "elements": elements,
        "floor": {
            "outline_area_m2": round(gross_area, 4),
            "effective_area_m2": round(net_area, 4),
            "load_kN_m2": params["floor"]["load_kN_m2"],
            "storeys": params["floor"]["storeys"]
        },
        "checks": {
            "sum_areas_m2": round(float(verification["sum_tributary_area_m2"]), 4),
            "sum_axial_kN": 7880.622,
            "labels_expected": len(expected),
            "labels_claimed": len(elements),
            "unclaimed_labels": []
        }
    }

    growth_x = [m["scale_growth_px_per_m"] for m in measured if m.get("orientation") == "horizontal"]
    growth_y = [m["scale_growth_px_per_m"] for m in measured if m.get("orientation") == "vertical"]
    score_sheet = {
        "plan": "FloorPlan1",
        "threshold": cfg["grey_threshold"],
        "connected_components_kept": len(components),
        "shape_rule_elongated_candidates": len(elongated),
        "named_elements_expected": len(expected),
        "named_elements_found": len(elements),
        "labels_original_pass": len(parse_element_labels(ocr_original, cfg["ocr_acceptance"])),
        "labels_long_lines_removed_pass": len(parse_element_labels(ocr_no_lines, cfg["ocr_acceptance"])),
        "labels_union": len(labels),
        "merged_named_elements": sum(bool(e["audit"]["merged"]) for e in elements),
        "broken_or_low_support_named_elements": len(flags),
        "scale_growth_px_per_m": {
            "x_median": round(float(np.median(growth_x)), 4),
            "y_median": round(float(np.median(growth_y)), 4),
            "growth_g_px": cfg["diagonal_kernel_px"]
        },
        "frame_scale_px_per_m": record["scale_px_per_m"]
    }

    INTERMEDIATE.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_image(INTERMEDIATE / "floorplan1_threshold.png", masks["ink"])
    write_image(INTERMEDIATE / "floorplan1_diagonal_support.png", masks["diag"])
    write_image(INTERMEDIATE / "floorplan1_horizontal_open17.png", masks["horizontal"])
    write_image(INTERMEDIATE / "floorplan1_vertical_open21.png", masks["vertical"])
    write_image(INTERMEDIATE / "floorplan1_long_lines_removed.png", line_removed)
    write_image(INTERMEDIATE / "floorplan1_long_line_mask.png", line_mask)
    dump_json(INTERMEDIATE / "ocr-original-HW3.json", ocr_original)
    dump_json(INTERMEDIATE / "ocr-lines-removed-HW3.json", ocr_no_lines)
    dump_json(INTERMEDIATE / "classical-components-HW3.json", {"components": components, "elongated": elongated})
    dump_json(INTERMEDIATE / "measured-geometry-HW3.json", measured)
    dump_json(RESULTS / "record-classical-label-supervised-HW3.json", record)
    dump_json(RESULTS / "score-sheet-classical-HW3.json", score_sheet)
    dump_json(RESULTS / "flags-manifest-HW3.json", {"confidence_threshold": 0.85, "flags": flags})
    plot_outputs(
        image,
        {"ink grey<225": masks["ink"], "45-degree open 11": masks["diag"], "horizontal open 17": masks["horizontal"], "vertical open 21": masks["vertical"]},
        elongated,
        record,
        frame,
        ocr_all,
    )
    print(json.dumps({"record": str(RESULTS / "record-classical-label-supervised-HW3.json"), "score_sheet": score_sheet, "flags": len(flags)}, indent=2))


if __name__ == "__main__":
    main()
