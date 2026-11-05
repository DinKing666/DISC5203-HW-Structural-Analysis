"""Primary hand calculation for CM, wall stiffness, CR and force shares."""

from __future__ import annotations

import csv
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from scipy.linalg import eigh

from structural_params import (
    DESIGN_DIRECTION,
    DESIGN_FORCE_KN,
    E_KN_M2,
    FLOOR_LOAD_KN_M2,
    G_KN_M2,
    GRAVITY_M_S2,
    MASS_REGIONS,
    POISSON,
    SHEAR_KAPPA,
    STOREY_HEIGHT_M,
    STOREYS,
)


ROOT = Path(__file__).resolve().parents[2]
PLAN_DIR = ROOT / "HW3/output/FloorPlan1"
PROMOTED_PATH = PLAN_DIR / "results/record-promoted-FloorPlan1-HW3.json"
CLASSICAL_PATH = PLAN_DIR / "results/record-classical-label-supervised-HW3.json"
REFERENCE_PATH = ROOT / "HW3/config/reference-manual-FloorPlan1-HW3.json"
COMPARISON_PATH = PLAN_DIR / "results/vlm-comparison-FloorPlan1-HW3.json"
GEOMETRY_PATH = ROOT / "HW1/config/plan1_geometry.json"
RESULT_PATH = PLAN_DIR / "results/hand-structural-FloorPlan1-HW3.json"
CSV_PATH = PLAN_DIR / "results/wall-stiffness-and-forces-FloorPlan1-HW3.csv"
FIGURE_PATH = PLAN_DIR / "figures/hw3_06_cm_cr_and_wall_forces.png"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def clean(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 9)
    if isinstance(value, np.floating):
        return round(float(value), 9)
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, list):
        return [clean(item) for item in value]
    if isinstance(value, tuple):
        return [clean(item) for item in value]
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    return value


def mass_properties() -> dict[str, Any]:
    regions: list[dict[str, float | str]] = []
    area = first_x = first_y = 0.0
    for name, sign, x0, x1, y0, y1 in MASS_REGIONS:
        width = x1 - x0
        height = y1 - y0
        signed_area = sign * width * height
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        area += signed_area
        first_x += signed_area * cx
        first_y += signed_area * cy
        regions.append(
            {
                "name": name,
                "sign": sign,
                "area_m2": signed_area,
                "centroid_x_m": cx,
                "centroid_y_m": cy,
                "width_m": width,
                "height_m": height,
            }
        )
    xm = first_x / area
    ym = first_y / area
    polar_area = 0.0
    for region in regions:
        local = region["area_m2"] * (
            region["width_m"] ** 2 + region["height_m"] ** 2
        ) / 12.0
        parallel = region["area_m2"] * (
            (region["centroid_x_m"] - xm) ** 2
            + (region["centroid_y_m"] - ym) ** 2
        )
        polar_area += local + parallel
    mass_per_storey = FLOOR_LOAD_KN_M2 * area / GRAVITY_M_S2
    return {
        "regions": regions,
        "effective_area_m2": area,
        "cm_m": [xm, ym],
        "polar_area_about_cm_m4": polar_area,
        "mass_per_storey_t": mass_per_storey,
        "five_storey_mass_t": mass_per_storey * STOREYS,
        "polar_mass_per_storey_t_m2": FLOOR_LOAD_KN_M2 / GRAVITY_M_S2 * polar_area,
    }


def wall_stiffness(length_m: float, thickness_m: float, shear: bool) -> float:
    inertia = thickness_m * length_m**3 / 12.0
    bending_flexibility = STOREY_HEIGHT_M**3 / (3.0 * E_KN_M2 * inertia)
    if not shear:
        return 1.0 / bending_flexibility
    shear_area = SHEAR_KAPPA * thickness_m * length_m
    shear_flexibility = STOREY_HEIGHT_M / (G_KN_M2 * shear_area)
    return 1.0 / (bending_flexibility + shear_flexibility)


def record_walls(record: dict[str, Any]) -> list[dict[str, Any]]:
    walls: list[dict[str, Any]] = []
    for element in record["elements"]:
        if element["class"] != "wall":
            continue
        thickness_m = element["size_mm"][0] / 1000.0
        length_m = element["size_mm"][1] / 1000.0
        walls.append(
            {
                "id": element["labels"][0],
                "direction": "x" if element["angle_deg"] == 0.0 else "y",
                "x_m": element["centre_m"][0],
                "y_m": element["centre_m"][1],
                "length_m": length_m,
                "thickness_m": thickness_m,
                "continuous": bool(element["continuous"]),
                "k_shear_kN_m": wall_stiffness(length_m, thickness_m, True),
                "k_bending_only_kN_m": wall_stiffness(length_m, thickness_m, False),
            }
        )
    return walls


def assemble(
    walls: list[dict[str, Any]], cm: tuple[float, float], shear: bool
) -> dict[str, Any]:
    xm, ym = cm
    matrix = np.zeros((3, 3), dtype=float)
    included: list[str] = []
    key = "k_shear_kN_m" if shear else "k_bending_only_kN_m"
    for wall in walls:
        if not wall["continuous"]:
            continue
        k = wall[key]
        if wall["direction"] == "x":
            vector = np.array([1.0, 0.0, -(wall["y_m"] - ym)])
        else:
            vector = np.array([0.0, 1.0, wall["x_m"] - xm])
        matrix += k * np.outer(vector, vector)
        included.append(wall["id"])
    flexibility = np.linalg.inv(matrix)
    cr_from_f = np.array(
        [
            xm - flexibility[2, 1] / flexibility[2, 2],
            ym + flexibility[2, 0] / flexibility[2, 2],
        ]
    )
    x_walls = [w for w in walls if w["continuous"] and w["direction"] == "x"]
    y_walls = [w for w in walls if w["continuous"] and w["direction"] == "y"]
    cr_weighted = np.array(
        [
            sum(w[key] * w["x_m"] for w in y_walls) / sum(w[key] for w in y_walls),
            sum(w[key] * w["y_m"] for w in x_walls) / sum(w[key] for w in x_walls),
        ]
    )
    if np.linalg.norm(cr_from_f - cr_weighted) > 1.0e-9:
        raise AssertionError("flexibility and weighted-centre CR disagree")
    return {
        "included_walls": included,
        "K_at_cm": matrix,
        "F_at_cm": flexibility,
        "cr_m": cr_from_f,
        "eccentricity_m": float(np.linalg.norm(cr_from_f - np.array(cm))),
        "direct_weighted_cr_m": cr_weighted,
    }


def with_continuity(
    base_walls: list[dict[str, Any]], states: dict[str, bool]
) -> list[dict[str, Any]]:
    result = deepcopy(base_walls)
    for wall in result:
        wall["continuous"] = states[wall["id"]]
    return result


def vlm_continuity_states(
    classical: dict[str, Any], comparison: dict[str, Any]
) -> dict[str, dict[str, bool]]:
    base = {
        item["labels"][0]: bool(item["continuous"])
        for item in classical["elements"]
        if item["class"] == "wall"
    }
    states: dict[str, dict[str, bool]] = {}
    result_files = {
        "smolvlm2-256m": PLAN_DIR / "results/vlm-smolvlm2-256m-FloorPlan1-HW3.json",
        "llava-onevision-0.5b": PLAN_DIR / "results/vlm-llava-onevision-0.5b-FloorPlan1-HW3.json",
    }
    score_by_model = {item["model_key"]: item for item in comparison["models"]}
    for model_key, path in result_files.items():
        model_states = dict(base)
        result = read_json(path)
        answer_by_flag = {item["flag_id"]: item for item in result["answers"]}
        for detail in score_by_model[model_key]["details"]:
            if detail["status"] != "resolved":
                continue
            model_states[detail["label"]] = bool(
                answer_by_flag[detail["flag_id"]]["parsed_answer"]["continuous"]
            )
        states[model_key] = model_states
    return states


def design_wall_forces(
    walls: list[dict[str, Any]], cm: tuple[float, float], cr: tuple[float, float]
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    xm, ym = cm
    xr, yr = cr
    active = [wall for wall in walls if wall["continuous"]]
    x_walls = [wall for wall in active if wall["direction"] == "x"]
    kx = sum(wall["k_shear_kN_m"] for wall in x_walls)
    torsional_stiffness = sum(
        wall["k_shear_kN_m"]
        * (
            (wall["y_m"] - yr) ** 2
            if wall["direction"] == "x"
            else (wall["x_m"] - xr) ** 2
        )
        for wall in active
    )
    torque = -(ym - yr) * DESIGN_FORCE_KN
    rotation = torque / torsional_stiffness
    rows: list[dict[str, Any]] = []
    sum_x = sum_y = moment = 0.0
    for wall in walls:
        k = wall["k_shear_kN_m"]
        if not wall["continuous"]:
            push = turn = total = 0.0
        elif wall["direction"] == "x":
            push = DESIGN_FORCE_KN * k / kx
            turn = -k * (wall["y_m"] - yr) * rotation
            total = push + turn
            sum_x += total
            moment += -(wall["y_m"] - yr) * total
        else:
            push = 0.0
            turn = k * (wall["x_m"] - xr) * rotation
            total = turn
            sum_y += total
            moment += (wall["x_m"] - xr) * total
        rows.append(
            {
                **wall,
                "push_share": k / kx if wall["continuous"] and wall["direction"] == "x" else 0.0,
                "turn_force_per_kNm": (
                    -k * (wall["y_m"] - yr) / torsional_stiffness
                    if wall["continuous"] and wall["direction"] == "x"
                    else k * (wall["x_m"] - xr) / torsional_stiffness
                    if wall["continuous"]
                    else 0.0
                ),
                "force_push_kN": push,
                "force_turn_kN": turn,
                "force_total_kN": total,
            }
        )
    return rows, {
        "design_force_kN": DESIGN_FORCE_KN,
        "direction": DESIGN_DIRECTION,
        "torque_about_cr_kNm": torque,
        "torsional_stiffness_about_cr_kN_m": torsional_stiffness,
        "rotation_rad": rotation,
        "equilibrium_sum_fx_kN": sum_x,
        "equilibrium_sum_fy_kN": sum_y,
        "equilibrium_moment_about_cr_kNm": moment,
    }


def push_turn_checks(
    matrix: np.ndarray, flexibility: np.ndarray, cm: tuple[float, float], cr: tuple[float, float]
) -> dict[str, Any]:
    xm, ym = cm
    xr, yr = cr
    unit_at_cm = np.array([1.0, 0.0, 0.0])
    unit_at_cr = np.array([1.0, 0.0, -(yr - ym)])
    q_cm = flexibility @ unit_at_cm
    q_cr = flexibility @ unit_at_cr
    return {
        "unit_x_at_cm_load": unit_at_cm,
        "unit_x_at_cm_displacement": q_cm,
        "unit_x_at_cr_load_about_cm": unit_at_cr,
        "unit_x_at_cr_displacement": q_cr,
        "rotation_at_cm_push_rad": q_cm[2],
        "rotation_at_cr_push_rad": q_cr[2],
        "matrix_times_cm_displacement": matrix @ q_cm,
    }


def modal_properties(
    matrix: np.ndarray, mass: dict[str, Any], cr: tuple[float, float]
) -> dict[str, Any]:
    mass_matrix = np.diag(
        [
            mass["mass_per_storey_t"],
            mass["mass_per_storey_t"],
            mass["polar_mass_per_storey_t_m2"],
        ]
    )
    eigenvalues, eigenvectors = eigh(matrix, mass_matrix)
    modes: list[dict[str, Any]] = []
    labels = ("x-translation", "y-translation", "torsion")
    for index, (eigenvalue, vector) in enumerate(zip(eigenvalues, eigenvectors.T), start=1):
        energy = np.diag(mass_matrix) * vector**2
        fractions = energy / energy.sum()
        modes.append(
            {
                "mode": index,
                "eigenvalue_rad2_s2": eigenvalue,
                "period_s": 2.0 * math.pi / math.sqrt(eigenvalue),
                "mass_normalized_vector_at_cm": vector,
                "kinetic_energy_fractions": {
                    labels[i]: fractions[i] for i in range(3)
                },
                "dominant_character": labels[int(np.argmax(fractions))],
            }
        )
    # Uncoupled reference periods are evaluated about CR, where translation/turn decouple.
    kx = matrix[0, 0]
    ky = matrix[1, 1]
    xr, yr = cr
    # Coordinate transformation from CR DOFs to CM DOFs.
    transform = np.array(
        [
            [1.0, 0.0, -(mass["cm_m"][1] - yr)],
            [0.0, 1.0, mass["cm_m"][0] - xr],
            [0.0, 0.0, 1.0],
        ]
    )
    k_cr = transform.T @ matrix @ transform
    j_cr = k_cr[2, 2]
    tx = 2.0 * math.pi * math.sqrt(mass["mass_per_storey_t"] / kx)
    ty = 2.0 * math.pi * math.sqrt(mass["mass_per_storey_t"] / ky)
    ttheta = 2.0 * math.pi * math.sqrt(
        mass["polar_mass_per_storey_t_m2"] / j_cr
    )
    return {
        "mass_matrix_at_cm": mass_matrix,
        "coupled_modes": modes,
        "K_transformed_to_cr": k_cr,
        "uncoupled_periods_s": {"Tx": tx, "Ty": ty, "Ttheta": ttheta},
        "period_ratio_Ttheta_over_Tx": ttheta / tx,
    }


def write_csv(rows: list[dict[str, Any]]) -> None:
    fields = [
        "id",
        "direction",
        "x_m",
        "y_m",
        "length_m",
        "thickness_m",
        "continuous",
        "k_shear_kN_m",
        "k_bending_only_kN_m",
        "push_share",
        "turn_force_per_kNm",
        "force_push_kN",
        "force_turn_kN",
        "force_total_kN",
    ]
    with CSV_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: clean(row[field]) for field in fields})


def draw_plan(
    geometry: dict[str, Any], walls: list[dict[str, Any]], mass: dict[str, Any],
    promoted_cr: tuple[float, float], reference_cr: tuple[float, float], rows: list[dict[str, Any]]
) -> None:
    fig, (ax, force_ax) = plt.subplots(1, 2, figsize=(13.2, 7.2), constrained_layout=True)
    ax.add_patch(Polygon(geometry["outline"], closed=True, facecolor="#eef4fa", edgecolor="#263238", lw=1.4))
    for addition in geometry["additions"]:
        ax.add_patch(Polygon(addition, closed=True, facecolor="#eef4fa", edgecolor="#263238", lw=1.4))
    for opening in geometry["openings"]:
        ax.add_patch(Polygon(opening, closed=True, facecolor="white", edgecolor="#607d8b", hatch="//", lw=1.0))
    for wall in walls:
        half = wall["length_m"] / 2.0
        if wall["direction"] == "x":
            xs = [wall["x_m"] - half, wall["x_m"] + half]
            ys = [wall["y_m"], wall["y_m"]]
        else:
            xs = [wall["x_m"], wall["x_m"]]
            ys = [wall["y_m"] - half, wall["y_m"] + half]
        color = "#c62828" if not wall["continuous"] else "#263238"
        ax.plot(xs, ys, color=color, lw=5.0, solid_capstyle="butt", ls="--" if not wall["continuous"] else "-")
        ax.text(wall["x_m"] + 0.08, wall["y_m"] + 0.08, wall["id"], fontsize=7.5)
    xm, ym = mass["cm_m"]
    ax.scatter([xm], [ym], marker="*", s=145, color="#1565c0", label="CM")
    ax.scatter([promoted_cr[0]], [promoted_cr[1]], marker="X", s=110, color="#c62828", label="CR promoted")
    ax.scatter([reference_cr[0]], [reference_cr[1]], marker="P", s=100, color="#2e7d32", label="CR reference")
    ax.plot([xm, promoted_cr[0]], [ym, promoted_cr[1]], color="#c62828", ls=":", lw=1.4)
    ax.set_title("Promoted extraction: CM and CR\nT3 dashed because the VLM gate left it unresolved")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, 11.4)
    ax.set_ylim(-0.5, 18.2)
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", fontsize=8)

    active_rows = [row for row in rows if row["continuous"]]
    ids = [row["id"] for row in active_rows]
    pushes = [row["force_push_kN"] for row in active_rows]
    turns = [row["force_turn_kN"] for row in active_rows]
    positions = np.arange(len(ids))
    force_ax.bar(positions, pushes, color="#64b5f6", label="push through CR")
    force_ax.bar(positions, turns, bottom=pushes, color="#ef5350", label="torsional turn")
    force_ax.axhline(0.0, color="#263238", lw=0.8)
    force_ax.set_xticks(positions, ids, rotation=45)
    force_ax.set_ylabel("wall force in its resisting direction (kN)")
    force_ax.set_title(f"{DESIGN_FORCE_KN:.0f} kN +x at CM: push + turn")
    force_ax.grid(axis="y", alpha=0.2)
    force_ax.legend(fontsize=8)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=180)
    plt.close(fig)


def main() -> None:
    promoted = read_json(PROMOTED_PATH)
    classical = read_json(CLASSICAL_PATH)
    reference = read_json(REFERENCE_PATH)
    comparison = read_json(COMPARISON_PATH)
    geometry = read_json(GEOMETRY_PATH)
    mass = mass_properties()
    cm = tuple(mass["cm_m"])
    promoted_walls = record_walls(promoted)
    shear = assemble(promoted_walls, cm, True)
    bending_only = assemble(promoted_walls, cm, False)

    reference_states = {
        wall["id"]: bool(reference["walls"][wall["id"]]["continuous"])
        for wall in promoted_walls
    }
    reference_walls = with_continuity(promoted_walls, reference_states)
    reference_shear = assemble(reference_walls, cm, True)

    model_states = vlm_continuity_states(classical, comparison)
    model_consequences: dict[str, Any] = {}
    for model_key, states in model_states.items():
        model_result = assemble(with_continuity(promoted_walls, states), cm, True)
        model_consequences[model_key] = {
            "cr_m": model_result["cr_m"],
            "eccentricity_m": model_result["eccentricity_m"],
            "cr_shift_from_reference_m": model_result["cr_m"] - reference_shear["cr_m"],
            "shift_norm_from_reference_m": float(
                np.linalg.norm(model_result["cr_m"] - reference_shear["cr_m"])
            ),
        }

    rows, design = design_wall_forces(promoted_walls, cm, tuple(shear["cr_m"]))
    checks = push_turn_checks(
        shear["K_at_cm"], shear["F_at_cm"], cm, tuple(shear["cr_m"])
    )
    modes = modal_properties(shear["K_at_cm"], mass, tuple(shear["cr_m"]))
    output = {
        "plan": "FloorPlan1",
        "primary_source": "record-promoted-FloorPlan1-HW3.json",
        "mechanics": {
            "E_kN_m2": E_KN_M2,
            "nu": POISSON,
            "G_kN_m2": G_KN_M2,
            "height_m": STOREY_HEIGHT_M,
            "shear_kappa": SHEAR_KAPPA,
            "wall_formula": "1 / (H^3/(3EI) + H/(G*(5/6)*t*L))",
            "continuous_false_excluded": True,
        },
        "mass": mass,
        "promoted_shear_included": shear,
        "promoted_bending_only": bending_only,
        "independent_reference_shear": reference_shear,
        "vlm_structural_consequence": model_consequences,
        "design_load_and_equilibrium": design,
        "push_turn_decoupling": checks,
        "modal_properties": modes,
        "checks": {
            "area_matches_promoted_m2": abs(
                mass["effective_area_m2"] - promoted["floor"]["effective_area_m2"]
            ),
            "cr_routes_difference_m": float(
                np.linalg.norm(shear["cr_m"] - shear["direct_weighted_cr_m"])
            ),
            "unit_push_at_cr_rotation_abs_rad": abs(checks["rotation_at_cr_push_rad"]),
            "design_force_residual_kN": design["equilibrium_sum_fx_kN"] - DESIGN_FORCE_KN,
            "cross_wall_force_residual_kN": design["equilibrium_sum_fy_kN"],
            "design_moment_residual_kNm": (
                design["equilibrium_moment_about_cr_kNm"]
                - design["torque_about_cr_kNm"]
            ),
        },
    }
    RESULT_PATH.write_text(
        json.dumps(clean(output), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_csv(rows)
    draw_plan(
        geometry,
        promoted_walls,
        mass,
        tuple(shear["cr_m"]),
        tuple(reference_shear["cr_m"]),
        rows,
    )
    print(
        f"CM=({cm[0]:.4f}, {cm[1]:.4f}) m; promoted CR="
        f"({shear['cr_m'][0]:.4f}, {shear['cr_m'][1]:.4f}) m; "
        f"e={shear['eccentricity_m']:.4f} m"
    )
    print(
        f"Reference CR=({reference_shear['cr_m'][0]:.4f}, "
        f"{reference_shear['cr_m'][1]:.4f}) m"
    )


if __name__ == "__main__":
    main()
