"""Hand re-analysis for the selected CM=CR wall modification."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from run_hand_structural import (
    assemble,
    clean,
    design_wall_forces,
    mass_properties,
    modal_properties,
    push_turn_checks,
    read_json,
    record_walls,
    wall_stiffness,
)
from structural_params import DESIGN_FORCE_KN


ROOT = Path(__file__).resolve().parents[2]
PLAN_DIR = ROOT / "HW3/output/FloorPlan1"
PROMOTED_PATH = PLAN_DIR / "results/record-promoted-FloorPlan1-HW3.json"
BASELINE_PATH = PLAN_DIR / "results/hand-structural-FloorPlan1-HW3.json"
GEOMETRY_PATH = ROOT / "HW1/config/plan1_geometry.json"
OUTPUT_PATH = PLAN_DIR / "results/modified-hand-FloorPlan1-HW3.json"
CSV_PATH = PLAN_DIR / "results/modified-wall-forces-FloorPlan1-HW3.csv"
FIGURE_PATH = PLAN_DIR / "figures/hw3_07_modification_before_after.png"

MODIFICATIONS = {
    "T3": {
        "action": "rebuild the unresolved wall as continuous at the same centreline",
        "old_thickness_m": 0.25,
        "new_thickness_m": 0.309,
        "old_continuous": False,
        "new_continuous": True,
        "centreline_moved_m": 0.0,
    },
    "T7": {
        "action": "thicken the existing shaft-edge wall at the same centreline",
        "old_thickness_m": 0.25,
        "new_thickness_m": 0.365,
        "old_continuous": True,
        "new_continuous": True,
        "centreline_moved_m": 0.0,
    },
}


def modified_walls() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    base = record_walls(read_json(PROMOTED_PATH))
    modified = [dict(wall) for wall in base]
    for wall in modified:
        if wall["id"] not in MODIFICATIONS:
            continue
        change = MODIFICATIONS[wall["id"]]
        wall["thickness_m"] = change["new_thickness_m"]
        wall["continuous"] = change["new_continuous"]
        wall["k_shear_kN_m"] = wall_stiffness(
            wall["length_m"], wall["thickness_m"], True
        )
        wall["k_bending_only_kN_m"] = wall_stiffness(
            wall["length_m"], wall["thickness_m"], False
        )
    return base, modified


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
        "push_share",
        "force_push_kN",
        "force_turn_kN",
        "force_total_kN",
    ]
    with CSV_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: clean(row[field]) for field in fields})


def draw_floor(ax: Any, geometry: dict[str, Any]) -> None:
    ax.add_patch(
        Polygon(
            geometry["outline"], closed=True, facecolor="#f5f8fb",
            edgecolor="#263238", lw=1.2
        )
    )
    for addition in geometry["additions"]:
        ax.add_patch(
            Polygon(addition, closed=True, facecolor="#f5f8fb", edgecolor="#263238", lw=1.2)
        )
    for opening in geometry["openings"]:
        ax.add_patch(
            Polygon(opening, closed=True, facecolor="white", edgecolor="#78909c", hatch="//", lw=0.9)
        )


def draw_walls(ax: Any, walls: list[dict[str, Any]], modified: bool) -> None:
    for wall in walls:
        half = wall["length_m"] / 2.0
        if wall["direction"] == "x":
            xs = [wall["x_m"] - half, wall["x_m"] + half]
            ys = [wall["y_m"], wall["y_m"]]
        else:
            xs = [wall["x_m"], wall["x_m"]]
            ys = [wall["y_m"] - half, wall["y_m"] + half]
        changed = modified and wall["id"] in MODIFICATIONS
        color = "#1565c0" if changed else "#37474f" if wall["continuous"] else "#c62828"
        linewidth = 3.0 + 7.0 * wall["thickness_m"]
        ax.plot(
            xs, ys, color=color, lw=linewidth, solid_capstyle="butt",
            ls="-" if wall["continuous"] else "--"
        )
        ax.text(wall["x_m"] + 0.08, wall["y_m"] + 0.08, wall["id"], fontsize=7)


def draw_figure(
    geometry: dict[str, Any], base_walls: list[dict[str, Any]],
    new_walls: list[dict[str, Any]], cm: tuple[float, float],
    base_cr: tuple[float, float], new_cr: tuple[float, float],
    baseline_forces: dict[str, float], new_rows: list[dict[str, Any]]
) -> None:
    fig = plt.figure(figsize=(14.0, 8.0), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.15])
    axes = [fig.add_subplot(grid[0, index]) for index in range(3)]
    for ax, walls, cr, title, is_modified in (
        (axes[0], base_walls, base_cr, "Before: promoted extraction", False),
        (axes[1], new_walls, new_cr, "After: T3 rebuilt, T7 thickened", True),
    ):
        draw_floor(ax, geometry)
        draw_walls(ax, walls, is_modified)
        ax.scatter([cm[0]], [cm[1]], marker="*", s=125, color="#1565c0", label="CM")
        ax.scatter([cr[0]], [cr[1]], marker="X", s=95, color="#c62828", label="CR")
        ax.plot([cm[0], cr[0]], [cm[1], cr[1]], color="#c62828", ls=":")
        ax.set_title(title)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_xlim(-0.5, 11.4)
        ax.set_ylim(-0.5, 18.2)
        ax.set_aspect("equal")
        ax.grid(alpha=0.18)
        ax.legend(loc="upper right", fontsize=8)

    ids = [row["id"] for row in new_rows if row["continuous"]]
    before = [baseline_forces.get(wall_id, 0.0) for wall_id in ids]
    after = [next(row["force_total_kN"] for row in new_rows if row["id"] == wall_id) for wall_id in ids]
    positions = np.arange(len(ids))
    width = 0.38
    axes[2].bar(positions - width / 2, before, width, label="before", color="#90a4ae")
    axes[2].bar(positions + width / 2, after, width, label="after", color="#1565c0")
    axes[2].axhline(0.0, color="#263238", lw=0.8)
    axes[2].set_xticks(positions, ids, rotation=45)
    axes[2].set_ylabel("wall force in resisting direction (kN)")
    axes[2].set_title(f"{DESIGN_FORCE_KN:.0f} kN +x at CM")
    axes[2].grid(axis="y", alpha=0.2)
    axes[2].legend()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=180)
    plt.close(fig)


def main() -> None:
    baseline = read_json(BASELINE_PATH)
    geometry = read_json(GEOMETRY_PATH)
    mass = mass_properties()
    cm = tuple(mass["cm_m"])
    base_walls, new_walls = modified_walls()
    result = assemble(new_walls, cm, True)
    rows, design = design_wall_forces(new_walls, cm, tuple(result["cr_m"]))
    decoupling = push_turn_checks(
        result["K_at_cm"], result["F_at_cm"], cm, tuple(result["cr_m"])
    )
    modes = modal_properties(result["K_at_cm"], mass, tuple(result["cr_m"]))

    with (PLAN_DIR / "results/wall-stiffness-and-forces-FloorPlan1-HW3.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        baseline_rows = list(csv.DictReader(stream))
    baseline_forces = {
        row["id"]: float(row["force_total_kN"]) for row in baseline_rows
    }
    baseline_push = {
        row["id"]: float(row["force_push_kN"]) for row in baseline_rows
    }
    baseline_amplification = {
        wall_id: baseline_forces[wall_id] / push
        for wall_id, push in baseline_push.items()
        if abs(push) > 1.0e-12
    }
    new_forces = {row["id"]: row["force_total_kN"] for row in rows}
    force_changes = {
        wall_id: new_forces.get(wall_id, 0.0) - force
        for wall_id, force in baseline_forces.items()
    }
    force_changes["T3"] = new_forces["T3"] - baseline_forces["T3"]
    baseline_cr = tuple(baseline["promoted_shear_included"]["cr_m"])
    baseline_e = baseline["promoted_shear_included"]["eccentricity_m"]
    torsional_radius_sq = (
        baseline["design_load_and_equilibrium"]["torsional_stiffness_about_cr_kN_m"]
        / baseline["promoted_shear_included"]["K_at_cm"][0][0]
    )
    output = {
        "plan": "FloorPlan1",
        "mass_layout_changed": False,
        "centreline_or_column_moved": False,
        "modifications": MODIFICATIONS,
        "mass": mass,
        "architectural_constraints": (
            "Keep every column, opening, wall centreline and the shaft grid fixed; rebuild T3 "
            "on its drawn line and thicken T7 toward the non-usable shaft side."
        ),
        "modified_shear": result,
        "design_load_and_equilibrium": design,
        "push_turn_decoupling": decoupling,
        "modal_properties": modes,
        "before_after": {
            "cm_m": list(cm),
            "cr_before_m": list(baseline_cr),
            "cr_after_m": result["cr_m"],
            "eccentricity_before_m": baseline_e,
            "eccentricity_after_m": result["eccentricity_m"],
            "eccentricity_reduction_percent": 100.0 * (1.0 - result["eccentricity_m"] / baseline_e),
            "baseline_most_torsion_amplified_wall": max(
                baseline_amplification, key=baseline_amplification.get
            ),
            "baseline_max_amplification_ratio": max(baseline_amplification.values()),
            "baseline_torsional_radius_squared_m2": torsional_radius_sq,
            "baseline_torsional_radius_m": math.sqrt(torsional_radius_sq),
            "peak_abs_wall_force_before_kN": max(abs(value) for value in baseline_forces.values()),
            "peak_abs_wall_force_after_kN": max(abs(value) for value in new_forces.values()),
            "wall_force_change_kN": force_changes,
            "period_ratio_Ttheta_over_Tx_before": baseline["modal_properties"]["period_ratio_Ttheta_over_Tx"],
            "period_ratio_Ttheta_over_Tx_after": modes["period_ratio_Ttheta_over_Tx"],
            "first_mode_before": baseline["modal_properties"]["coupled_modes"][0],
            "first_mode_after": modes["coupled_modes"][0],
        },
        "checks": {
            "cm_x_minus_cr_x_m": cm[0] - result["cr_m"][0],
            "cm_y_minus_cr_y_m": cm[1] - result["cr_m"][1],
            "unit_push_at_cr_rotation_abs_rad": abs(decoupling["rotation_at_cr_push_rad"]),
            "design_force_residual_kN": design["equilibrium_sum_fx_kN"] - DESIGN_FORCE_KN,
            "cross_wall_force_residual_kN": design["equilibrium_sum_fy_kN"],
            "design_moment_residual_kNm": design["equilibrium_moment_about_cr_kNm"]
            - design["torque_about_cr_kNm"],
        },
    }
    OUTPUT_PATH.write_text(
        json.dumps(clean(output), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_csv(rows)
    draw_figure(
        geometry, base_walls, new_walls, cm, baseline_cr, tuple(result["cr_m"]),
        baseline_forces, rows
    )
    print(
        f"Modified CR=({result['cr_m'][0]:.6f}, {result['cr_m'][1]:.6f}) m; "
        f"CM=({cm[0]:.6f}, {cm[1]:.6f}) m; e={result['eccentricity_m']:.6f} m"
    )
    print(
        f"T3 thickness={MODIFICATIONS['T3']['new_thickness_m']:.3f} m; "
        f"T7 thickness={MODIFICATIONS['T7']['new_thickness_m']:.3f} m"
    )


if __name__ == "__main__":
    main()
