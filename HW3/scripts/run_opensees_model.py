"""OpenSees check of the frozen hand rigid-diaphragm calculation.

Run with the repository's Python 3.12 wrapper because the Windows OpenSees
extension is compiled against python312.dll.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / ".opensees-py312"
if sys.version_info[:2] != (3, 12):
    raise RuntimeError("OpenSees Windows binary requires Python 3.12")
sys.path.insert(0, str(TARGET))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import openseespy.opensees as ops  # noqa: E402

from structural_params import (  # noqa: E402
    DESIGN_FORCE_KN,
    E_KN_M2,
    G_KN_M2,
    SHEAR_KAPPA,
    STOREY_HEIGHT_M,
)


PROMOTED_PATH = ROOT / "HW3/output/FloorPlan1/results/record-promoted-FloorPlan1-HW3.json"
HAND_PATH = ROOT / "HW3/output/FloorPlan1/results/hand-structural-FloorPlan1-HW3.json"
OUTPUT_PATH = ROOT / "HW3/output/FloorPlan1/results/opensees-FloorPlan1-HW3.json"
MASTER_NODE = 9000
WEAK_RATIO = 1.0e-9


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def clean(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 12)
    if isinstance(value, list):
        return [clean(item) for item in value]
    if isinstance(value, tuple):
        return [clean(item) for item in value]
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    return value


def inverse_3x3(matrix: list[list[float]]) -> list[list[float]]:
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if abs(determinant) < 1.0e-30:
        raise RuntimeError("singular 3x3 matrix")
    return [
        [(e * i - f * h) / determinant, (c * h - b * i) / determinant, (b * f - c * e) / determinant],
        [(f * g - d * i) / determinant, (a * i - c * g) / determinant, (c * d - a * f) / determinant],
        [(d * h - e * g) / determinant, (b * g - a * h) / determinant, (a * e - b * d) / determinant],
    ]


def multiply(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [sum(row[j] * vector[j] for j in range(3)) for row in matrix]


def subtract(a: list[float], b: list[float]) -> list[float]:
    return [left - right for left, right in zip(a, b)]


def wall_records() -> list[dict[str, Any]]:
    record = read_json(PROMOTED_PATH)
    walls: list[dict[str, Any]] = []
    for element in record["elements"]:
        if element["class"] != "wall" or not element["continuous"]:
            continue
        walls.append(
            {
                "id": element["labels"][0],
                "direction": "x" if element["angle_deg"] == 0.0 else "y",
                "x_m": float(element["centre_m"][0]),
                "y_m": float(element["centre_m"][1]),
                "length_m": element["size_mm"][1] / 1000.0,
                "thickness_m": element["size_mm"][0] / 1000.0,
            }
        )
    return walls


def build_model(walls: list[dict[str, Any]], cm: tuple[float, float]) -> dict[str, dict[str, int]]:
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 6)
    xm, ym = cm
    ops.node(MASTER_NODE, xm, ym, STOREY_HEIGHT_M)
    ops.fix(MASTER_NODE, 0, 0, 1, 1, 1, 0)
    # Element local x is vertical.  These vectors make local y follow the wall length.
    ops.geomTransf("Linear", 1, 0.0, 1.0, 0.0)   # x-plan wall: local y = global x
    ops.geomTransf("Linear", 2, -1.0, 0.0, 0.0)  # y-plan wall: local y = global y
    tags: dict[str, dict[str, int]] = {}
    top_nodes: list[int] = []
    for index, wall in enumerate(walls, start=1):
        base_node = 100 + 2 * index
        top_node = base_node + 1
        element_tag = 1000 + index
        ops.node(base_node, wall["x_m"], wall["y_m"], 0.0)
        ops.node(top_node, wall["x_m"], wall["y_m"], STOREY_HEIGHT_M)
        ops.fix(base_node, 1, 1, 1, 1, 1, 1)
        length = wall["length_m"]
        thickness = wall["thickness_m"]
        area = length * thickness
        strong_i = thickness * length**3 / 12.0
        strong_av = SHEAR_KAPPA * area
        weak_i = strong_i * WEAK_RATIO
        weak_av = strong_av * WEAK_RATIO
        torsion_j = strong_i * WEAK_RATIO
        transf_tag = 1 if wall["direction"] == "x" else 2
        ops.element(
            "ElasticTimoshenkoBeam",
            element_tag,
            base_node,
            top_node,
            E_KN_M2,
            G_KN_M2,
            area,
            torsion_j,
            weak_i,
            strong_i,
            strong_av,
            weak_av,
            transf_tag,
        )
        tags[wall["id"]] = {
            "base_node": base_node,
            "top_node": top_node,
            "element": element_tag,
        }
        top_nodes.append(top_node)
    ops.rigidDiaphragm(3, MASTER_NODE, *top_nodes)
    return tags


def analyze_case(
    name: str,
    load: tuple[float, float, float],
    walls: list[dict[str, Any]],
    cm: tuple[float, float],
) -> dict[str, Any]:
    tags = build_model(walls, cm)
    ops.timeSeries("Constant", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(MASTER_NODE, load[0], load[1], 0.0, 0.0, 0.0, load[2])
    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("BandGeneral")
    ops.test("NormDispIncr", 1.0e-14, 20, 0)
    ops.algorithm("Linear")
    ops.integrator("LoadControl", 1.0)
    ops.analysis("Static")
    status = ops.analyze(1)
    if status != 0:
        raise RuntimeError(f"OpenSees failed in load case {name}: {status}")
    displacement = [
        float(ops.nodeDisp(MASTER_NODE, 1)),
        float(ops.nodeDisp(MASTER_NODE, 2)),
        float(ops.nodeDisp(MASTER_NODE, 6)),
    ]
    wall_forces: dict[str, Any] = {}
    for wall in walls:
        tag = tags[wall["id"]]["element"]
        local = [float(value) for value in ops.eleResponse(tag, "localForce")]
        if len(local) != 12:
            raise RuntimeError(f"unexpected local-force vector for {wall['id']}: {local}")
        wall_forces[wall["id"]] = {
            "direction": wall["direction"],
            "strong_top_shear_kN": local[7],
            "local_force_12": local,
        }
    return {
        "name": name,
        "load_at_cm_dofs": list(load),
        "master_displacement": displacement,
        "wall_forces": wall_forces,
    }


def main() -> None:
    hand = read_json(HAND_PATH)
    cm = tuple(float(value) for value in hand["mass"]["cm_m"])
    hand_cr = tuple(float(value) for value in hand["promoted_shear_included"]["cr_m"])
    walls = wall_records()
    unit_cases = [
        analyze_case("unit Px at CM", (1.0, 0.0, 0.0), walls, cm),
        analyze_case("unit Py at CM", (0.0, 1.0, 0.0), walls, cm),
        analyze_case("unit Mz at CM", (0.0, 0.0, 1.0), walls, cm),
    ]
    flexibility = [
        [unit_cases[column]["master_displacement"][row] for column in range(3)]
        for row in range(3)
    ]
    stiffness = inverse_3x3(flexibility)
    model_cr = (
        cm[0] - flexibility[2][1] / flexibility[2][2],
        cm[1] + flexibility[2][0] / flexibility[2][2],
    )
    eccentricity = math.hypot(model_cr[0] - cm[0], model_cr[1] - cm[1])

    y_offset = model_cr[1] - cm[1]
    design_total = analyze_case(
        "1000 kN +x at CM", (DESIGN_FORCE_KN, 0.0, 0.0), walls, cm
    )
    design_push = analyze_case(
        "1000 kN +x through CR",
        (DESIGN_FORCE_KN, 0.0, -y_offset * DESIGN_FORCE_KN),
        walls,
        cm,
    )
    design_turn = analyze_case(
        "CM eccentric torque about CR",
        (0.0, 0.0, y_offset * DESIGN_FORCE_KN),
        walls,
        cm,
    )
    unit_cr = analyze_case(
        "unit +x through CR", (1.0, 0.0, -y_offset), walls, cm
    )

    additivity: dict[str, float] = {}
    for wall in walls:
        wall_id = wall["id"]
        total = design_total["wall_forces"][wall_id]["strong_top_shear_kN"]
        parts = (
            design_push["wall_forces"][wall_id]["strong_top_shear_kN"]
            + design_turn["wall_forces"][wall_id]["strong_top_shear_kN"]
        )
        additivity[wall_id] = total - parts

    hand_k = hand["promoted_shear_included"]["K_at_cm"]
    k_difference = [
        [stiffness[row][column] - hand_k[row][column] for column in range(3)]
        for row in range(3)
    ]
    k_relative = [
        [
            k_difference[row][column] / hand_k[row][column]
            if abs(hand_k[row][column]) > 1.0e-9
            else k_difference[row][column]
            for column in range(3)
        ]
        for row in range(3)
    ]
    total_wall_x = sum(
        design_total["wall_forces"][wall["id"]]["strong_top_shear_kN"]
        for wall in walls
        if wall["direction"] == "x"
    )
    total_wall_y = sum(
        design_total["wall_forces"][wall["id"]]["strong_top_shear_kN"]
        for wall in walls
        if wall["direction"] == "y"
    )
    output = {
        "plan": "FloorPlan1",
        "runtime": {
            "python": sys.version.split()[0],
            "opensees": ops.version(),
            "isolated_target": ".opensees-py312",
        },
        "model": {
            "ndm": 3,
            "ndf": 6,
            "element": "ElasticTimoshenkoBeam",
            "one_vertical_element_per_continuous_wall": True,
            "height_m": STOREY_HEIGHT_M,
            "strong_axis_I": "t*L^3/12",
            "strong_axis_Av": "(5/6)*t*L",
            "weak_axis_and_st_venant_torsion_ratio": WEAK_RATIO,
            "rigid_diaphragm": "rigidDiaphragm(3, master, *wall_top_nodes)",
            "constraint_handler": "Transformation",
            "master_free_dofs": ["ux", "uy", "rz"],
            "included_walls": [wall["id"] for wall in walls],
        },
        "unit_load_cases": unit_cases,
        "flexibility_at_cm": flexibility,
        "stiffness_at_cm": stiffness,
        "cr_from_flexibility_m": list(model_cr),
        "eccentricity_m": eccentricity,
        "design_cases": {
            "total_at_cm": design_total,
            "push_through_cr": design_push,
            "turn_about_cr": design_turn,
            "unit_push_through_cr": unit_cr,
        },
        "comparison_to_hand": {
            "hand_cr_m": list(hand_cr),
            "cr_difference_m": subtract(list(model_cr), list(hand_cr)),
            "cr_difference_norm_m": math.hypot(
                model_cr[0] - hand_cr[0], model_cr[1] - hand_cr[1]
            ),
            "K_model_minus_hand": k_difference,
            "K_relative_difference": k_relative,
            "max_abs_K_relative_difference": max(
                abs(value) for row in k_relative for value in row
            ),
            "unit_push_through_cr_rotation_rad": unit_cr["master_displacement"][2],
            "design_wall_sum_x_kN": total_wall_x,
            "design_cross_wall_sum_y_kN": total_wall_y,
            "max_wall_force_additivity_residual_kN": max(
                abs(value) for value in additivity.values()
            ),
            "wall_force_additivity_residual_kN": additivity,
        },
    }
    OUTPUT_PATH.write_text(
        json.dumps(clean(output), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"OpenSees {ops.version()} CR=({model_cr[0]:.6f}, {model_cr[1]:.6f}) m; "
        f"hand difference={output['comparison_to_hand']['cr_difference_norm_m']:.6e} m"
    )
    print(
        f"K max relative difference="
        f"{output['comparison_to_hand']['max_abs_K_relative_difference']:.3e}; "
        f"design sums=({total_wall_x:.6f}, {total_wall_y:.6f}) kN"
    )


if __name__ == "__main__":
    main()
