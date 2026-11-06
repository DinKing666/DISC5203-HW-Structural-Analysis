"""Hand CM/CR calculation for the course-supplied NewPlan generalisation test."""

from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from run_hand_structural import (
    assemble,
    clean,
    design_wall_forces,
    modal_properties,
    push_turn_checks,
    record_walls,
    wall_stiffness,
    with_continuity,
)
from structural_params import (
    DESIGN_DIRECTION,
    DESIGN_FORCE_KN,
    E_KN_M2,
    FLOOR_LOAD_KN_M2,
    G_KN_M2,
    GRAVITY_M_S2,
    POISSON,
    SHEAR_KAPPA,
    STOREY_HEIGHT_M,
    STOREYS,
)


ROOT = Path(__file__).resolve().parents[2]
PLAN_DIR = ROOT / "HW3/output/NewPlan"
PROMOTED_PATH = PLAN_DIR / "results/record-promoted-NewPlan-HW3.json"
CLASSICAL_PATH = (
    PLAN_DIR / "results/record-classical-label-supervised-NewPlan-HW3.json"
)
REFERENCE_PATH = ROOT / "HW3/config/reference-manual-NewPlan-HW3.json"
COMPARISON_PATH = PLAN_DIR / "results/vlm-comparison-NewPlan-HW3.json"
RESULT_PATH = PLAN_DIR / "results/hand-structural-NewPlan-HW3.json"
CSV_PATH = PLAN_DIR / "results/wall-stiffness-and-forces-NewPlan-HW3.csv"
FIGURE_PATH = PLAN_DIR / "figures/hw3_13_newplan_cm_cr_and_wall_forces.png"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mass_properties() -> dict[str, Any]:
    width = 46.0
    height = 18.0
    area = width * height
    polar_area = area * (width**2 + height**2) / 12.0
    mass_per_storey = FLOOR_LOAD_KN_M2 * area / GRAVITY_M_S2
    return {
        "regions": [
            {
                "name": "46 m x 18 m floor rectangle",
                "sign": 1.0,
                "area_m2": area,
                "centroid_x_m": width / 2.0,
                "centroid_y_m": height / 2.0,
                "width_m": width,
                "height_m": height,
            }
        ],
        "effective_area_m2": area,
        "cm_m": [width / 2.0, height / 2.0],
        "polar_area_about_cm_m4": polar_area,
        "mass_per_storey_t": mass_per_storey,
        "five_storey_mass_t": mass_per_storey * STOREYS,
        "polar_mass_per_storey_t_m2": FLOOR_LOAD_KN_M2 / GRAVITY_M_S2 * polar_area,
    }


def reference_walls(reference: dict[str, Any]) -> list[dict[str, Any]]:
    walls: list[dict[str, Any]] = []
    for wall_id, item in reference["walls"].items():
        dimensions = [int(value) for value in item["size_mm"].lower().split("x")]
        thickness_m = min(dimensions) / 1000.0
        length_m = max(dimensions) / 1000.0
        walls.append(
            {
                "id": wall_id,
                "direction": "x" if item["angle_deg"] == 0.0 else "y",
                "x_m": item["centre_m"][0],
                "y_m": item["centre_m"][1],
                "length_m": length_m,
                "thickness_m": thickness_m,
                "continuous": bool(item["continuous"]),
                "k_shear_kN_m": wall_stiffness(length_m, thickness_m, True),
                "k_bending_only_kN_m": wall_stiffness(length_m, thickness_m, False),
            }
        )
    return walls


def vlm_continuity_states(
    classical: dict[str, Any], comparison: dict[str, Any]
) -> dict[str, dict[str, bool]]:
    base = {
        item["labels"][0]: bool(item["continuous"])
        for item in classical["elements"]
        if item["class"] == "wall"
    }
    result_files = {
        "smolvlm2-256m": PLAN_DIR
        / "results/vlm-smolvlm2-256m-NewPlan-HW3.json",
        "llava-onevision-0.5b": PLAN_DIR
        / "results/vlm-llava-onevision-0.5b-NewPlan-HW3.json",
    }
    score_by_model = {item["model_key"]: item for item in comparison["models"]}
    states: dict[str, dict[str, bool]] = {}
    for model_key, path in result_files.items():
        model_states = dict(base)
        result = read_json(path)
        answer_by_flag = {item["flag_id"]: item for item in result["answers"]}
        for detail in score_by_model[model_key]["details"]:
            if detail["status"] != "resolved":
                continue
            answer = answer_by_flag[detail["flag_id"]]["parsed_answer"]
            model_states[detail["label"]] = bool(answer["continuous"])
        states[model_key] = model_states
    return states


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
    walls: list[dict[str, Any]],
    mass: dict[str, Any],
    promoted_cr: tuple[float, float],
    reference_cr: tuple[float, float],
    rows: list[dict[str, Any]],
) -> None:
    fig, (axis, force_axis) = plt.subplots(
        2, 1, figsize=(13.5, 9.2), gridspec_kw={"height_ratios": [2.1, 1.0]},
        constrained_layout=True,
    )
    axis.add_patch(
        Rectangle((0.0, 0.0), 46.0, 18.0, facecolor="#eef4fa", edgecolor="#263238", lw=1.4)
    )
    for wall in walls:
        half = wall["length_m"] / 2.0
        if wall["direction"] == "x":
            xs = [wall["x_m"] - half, wall["x_m"] + half]
            ys = [wall["y_m"], wall["y_m"]]
        else:
            xs = [wall["x_m"], wall["x_m"]]
            ys = [wall["y_m"] - half, wall["y_m"] + half]
        axis.plot(xs, ys, color="#263238", lw=5.0, solid_capstyle="butt")
        axis.text(wall["x_m"] + 0.25, wall["y_m"] + 0.25, wall["id"], fontsize=8)
    xm, ym = mass["cm_m"]
    axis.scatter([xm], [ym], marker="*", s=160, color="#1565c0", label="CM")
    axis.scatter(
        [promoted_cr[0]], [promoted_cr[1]], marker="X", s=115,
        color="#c62828", label="CR promoted extraction",
    )
    axis.scatter(
        [reference_cr[0]], [reference_cr[1]], marker="P", s=105,
        color="#2e7d32", label="CR independent reference",
    )
    axis.plot([xm, promoted_cr[0]], [ym, promoted_cr[1]], color="#c62828", ls=":", lw=1.4)
    axis.set_title("NewPlan hand calculation: extracted wall centres versus frozen reference")
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_aspect("equal")
    axis.set_xlim(-1.0, 47.0)
    axis.set_ylim(-1.0, 19.0)
    axis.grid(alpha=0.2)
    axis.legend(loc="upper center", ncol=3, fontsize=8)

    ids = [row["id"] for row in rows if row["continuous"]]
    pushes = [row["force_push_kN"] for row in rows if row["continuous"]]
    turns = [row["force_turn_kN"] for row in rows if row["continuous"]]
    positions = np.arange(len(ids))
    force_axis.bar(positions, pushes, color="#64b5f6", label="push through CR")
    force_axis.bar(positions, turns, bottom=pushes, color="#ef5350", label="torsional turn")
    force_axis.axhline(0.0, color="#263238", lw=0.8)
    force_axis.set_xticks(positions, ids)
    force_axis.set_ylabel("wall force (kN)")
    force_axis.set_title(f"{DESIGN_FORCE_KN:.0f} kN +x at CM: push + turn")
    force_axis.grid(axis="y", alpha=0.2)
    force_axis.legend(fontsize=8)
    fig.savefig(FIGURE_PATH, dpi=180)
    plt.close(fig)


def main() -> None:
    promoted = read_json(PROMOTED_PATH)
    classical = read_json(CLASSICAL_PATH)
    reference = read_json(REFERENCE_PATH)
    comparison = read_json(COMPARISON_PATH)
    mass = mass_properties()
    cm = tuple(mass["cm_m"])

    promoted_walls = record_walls(promoted)
    promoted_shear = assemble(promoted_walls, cm, True)
    promoted_bending = assemble(promoted_walls, cm, False)
    reference_shear = assemble(reference_walls(reference), cm, True)

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
            "continuity_states": states,
        }

    rows, design = design_wall_forces(
        promoted_walls, cm, tuple(promoted_shear["cr_m"])
    )
    checks = push_turn_checks(
        promoted_shear["K_at_cm"],
        promoted_shear["F_at_cm"],
        cm,
        tuple(promoted_shear["cr_m"]),
    )
    modes = modal_properties(
        promoted_shear["K_at_cm"], mass, tuple(promoted_shear["cr_m"])
    )
    output = {
        "plan": "NewPlan",
        "primary_source": "record-promoted-NewPlan-HW3.json",
        "mechanics": {
            "E_kN_m2": E_KN_M2,
            "nu": POISSON,
            "G_kN_m2": G_KN_M2,
            "height_m": STOREY_HEIGHT_M,
            "shear_kappa": SHEAR_KAPPA,
            "wall_formula": "1 / (H^3/(3EI) + H/(G*(5/6)*t*L))",
            "continuous_false_excluded": True,
            "locked_to_floorplan1_mechanics": True,
        },
        "mass": mass,
        "promoted_shear_included": promoted_shear,
        "promoted_bending_only": promoted_bending,
        "independent_reference_shear": reference_shear,
        "vlm_structural_consequence": model_consequences,
        "design_load_and_equilibrium": design,
        "push_turn_decoupling": checks,
        "modal_properties": modes,
        "checks": {
            "area_matches_promoted_m2": abs(
                mass["effective_area_m2"] - promoted["floor"]["effective_area_m2"]
            ),
            "cm_matches_reference_m": float(
                np.linalg.norm(np.array(cm) - np.array(reference["floor"]["centroid_m"]))
            ),
            "cr_routes_difference_m": float(
                np.linalg.norm(
                    promoted_shear["cr_m"] - promoted_shear["direct_weighted_cr_m"]
                )
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
        promoted_walls,
        mass,
        tuple(promoted_shear["cr_m"]),
        tuple(reference_shear["cr_m"]),
        rows,
    )
    print(
        f"CM=({cm[0]:.4f}, {cm[1]:.4f}) m; promoted CR="
        f"({promoted_shear['cr_m'][0]:.4f}, {promoted_shear['cr_m'][1]:.4f}) m; "
        f"e={promoted_shear['eccentricity_m']:.4f} m"
    )
    print(
        f"Reference CR=({reference_shear['cr_m'][0]:.4f}, "
        f"{reference_shear['cr_m'][1]:.4f}) m; "
        f"e={reference_shear['eccentricity_m']:.4f} m"
    )


if __name__ == "__main__":
    main()
