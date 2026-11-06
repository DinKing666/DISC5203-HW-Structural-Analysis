"""Freeze Task 8 independent references before either VLM sees a crop."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from shapely.geometry import Polygon


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "generalization-plans-HW3.json"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def new_plan_reference(config: dict) -> dict:
    plan = config["plans"]["NewPlan"]
    polygon = Polygon(plan["floor_polygon_m"])
    columns = {
        item["id"]: {
            "centre_m": item["point"],
            "size_mm": plan["printed_column_size_mm"],
            "continuous": True,
        }
        for item in plan["columns"]
    }
    walls = {}
    for item in plan["walls"]:
        label = item["id"]
        start, end = item["start"], item["end"]
        size = plan["printed_wall_sizes_mm"][label]
        walls[label] = {
            "centre_m": [
                round((start[0] + end[0]) / 2, 6),
                round((start[1] + end[1]) / 2, 6),
            ],
            "size_mm": f"{size[0]}x{size[1]}",
            "angle_deg": 0.0 if start[1] == end[1] else 90.0,
            "continuous": True,
        }
    return {
        "plan": "NewPlan",
        "built_before_vlm": True,
        "reference_route": (
            "Manual reading of the printed grid, C1:C30 global column note and each "
            "T1:T8 size callout; independent of the locked morphology and both VLMs."
        ),
        "source_sha256": plan["sha256"],
        "floor": {
            "polygon_m": plan["floor_polygon_m"],
            "area_m2": round(polygon.area, 6),
            "centroid_m": [round(polygon.centroid.x, 6), round(polygon.centroid.y, 6)],
        },
        "columns": columns,
        "walls": walls,
    }


def mlstruct_reference(config: dict) -> dict:
    plan = config["plans"]["MLStructFP966"]
    annotation_path = ROOT / plan["annotation_source"]
    source_path = ROOT / plan["source"]
    license_path = ROOT / plan["license"]
    for path, expected in [
        (source_path, plan["sha256"]),
        (annotation_path, plan["annotation_sha256"]),
        (license_path, plan["license_sha256"]),
    ]:
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"Frozen source hash changed: {path}: {actual}")

    data = read_json(annotation_path)
    floor_id = plan["dataset_floor_id"]
    floor = data["floor"][str(floor_id)]
    slabs = [
        (key, value)
        for key, value in data["slab"].items()
        if value["floorID"] == floor_id
    ]
    if len(slabs) != 1:
        raise RuntimeError(f"Expected one slab for floor {floor_id}, got {len(slabs)}")
    slab_id, slab = slabs[0]
    polygon = Polygon(list(zip(slab["x"], slab["y"])))
    rects = []
    for rect_id, item in data["rect"].items():
        if item["floorID"] != floor_id:
            continue
        centre = [round(sum(item["x"]) / 4, 6), round(sum(item["y"]) / 4, 6)]
        rects.append(
            {
                "label": f"R{rect_id}",
                "dataset_rect_id": int(rect_id),
                "dataset_wall_id": item["wallID"],
                "centre_m": centre,
                "size_mm": [
                    int(round(item["thickness"] * 1000)),
                    int(round(item["length"] * 1000)),
                ],
                "angle_deg": item["angle"],
                "continuous": True,
                "printed_label_present": False,
            }
        )
    rects.sort(key=lambda item: item["dataset_rect_id"])
    return {
        "plan": "MLStructFP966",
        "built_before_vlm": True,
        "reference_route": (
            "Public MLStructFP test annotation for dataset floor 966, selected before "
            "either VLM run; the JSON polygons and scale are independent of the locked "
            "raster morphology. They are a defensible dataset reference, not ground truth."
        ),
        "source_url": plan["source_url"],
        "source_sha256": plan["sha256"],
        "annotation_sha256": plan["annotation_sha256"],
        "license": "MIT",
        "dataset_floor_id": floor_id,
        "dataset_image_name": floor["image"],
        "scale_px_per_m": floor["scale"],
        "floor": {
            "dataset_slab_id": int(slab_id),
            "polygon_m": list(zip(slab["x"], slab["y"])),
            "area_m2": round(polygon.area, 6),
            "centroid_m": [round(polygon.centroid.x, 6), round(polygon.centroid.y, 6)],
        },
        "wall_rectangles": rects,
        "wall_group_count": len({item["dataset_wall_id"] for item in rects}),
        "printed_element_size_labels": 0,
    }


def main() -> None:
    config = read_json(CONFIG)
    write_json(
        ROOT / "config" / "reference-manual-NewPlan-HW3.json",
        new_plan_reference(config),
    )
    write_json(
        ROOT / "config" / "reference-dataset-MLStructFP966-HW3.json",
        mlstruct_reference(config),
    )
    print("Frozen Task 8 references for NewPlan and MLStructFP966")


if __name__ == "__main__":
    main()
