"""Reference-only CM/CR benchmark for MLStructFP966 after extraction fails safely."""

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

from run_hand_structural import clean, wall_stiffness
from structural_params import (
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
PLAN_DIR = ROOT / "HW3/output/MLStructFP966"
REFERENCE_PATH = ROOT / "HW3/config/reference-dataset-MLStructFP966-HW3.json"
RESULT_PATH = (
    PLAN_DIR / "results/reference-structural-benchmark-MLStructFP966-HW3.json"
)
CSV_PATH = PLAN_DIR / "results/reference-wall-stiffness-MLStructFP966-HW3.csv"
FIGURE_PATH = PLAN_DIR / "figures/hw3_19_mlstruct_reference_cm_cr.png"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def polygon_mass_properties(points: list[list[float]]) -> dict[str, Any]:
    vertices = np.asarray(points, dtype=float)
    following = np.roll(vertices, -1, axis=0)
    cross = vertices[:, 0] * following[:, 1] - following[:, 0] * vertices[:, 1]
    signed_area = 0.5 * float(cross.sum())
    cx = float(((vertices[:, 0] + following[:, 0]) * cross).sum() / (6.0 * signed_area))
    cy = float(((vertices[:, 1] + following[:, 1]) * cross).sum() / (6.0 * signed_area))
    ix_origin = float(
        (
            (
                vertices[:, 1] ** 2
                + vertices[:, 1] * following[:, 1]
                + following[:, 1] ** 2
            )
            * cross
        ).sum()
        / 12.0
    )
    iy_origin = float(
        (
            (
                vertices[:, 0] ** 2
                + vertices[:, 0] * following[:, 0]
                + following[:, 0] ** 2
            )
            * cross
        ).sum()
        / 12.0
    )
    sign = 1.0 if signed_area > 0 else -1.0
    area = abs(signed_area)
    polar_area = sign * (ix_origin + iy_origin) - area * (cx**2 + cy**2)
    mass_per_storey = FLOOR_LOAD_KN_M2 * area / GRAVITY_M_S2
    return {
        "effective_area_m2": area,
        "cm_m": [cx, cy],
        "polar_area_about_cm_m4": polar_area,
        "mass_per_storey_t": mass_per_storey,
        "five_storey_mass_t": mass_per_storey * STOREYS,
        "polar_mass_per_storey_t_m2": FLOOR_LOAD_KN_M2 / GRAVITY_M_S2 * polar_area,
    }


def wall_rows(reference: dict[str, Any], cm: tuple[float, float]) -> list[dict[str, Any]]:
    xm, ym = cm
    rows: list[dict[str, Any]] = []
    for item in reference["wall_rectangles"]:
        thickness_m = item["size_mm"][0] / 1000.0
        length_m = item["size_mm"][1] / 1000.0
        angle_rad = math.radians(item["angle_deg"])
        dx, dy = math.cos(angle_rad), math.sin(angle_rad)
        x, y = item["centre_m"]
        lever = -dx * (y - ym) + dy * (x - xm)
        rows.append(
            {
                "id": item["label"],
                "dataset_wall_id": item["dataset_wall_id"],
                "x_m": x,
                "y_m": y,
                "length_m": length_m,
                "thickness_m": thickness_m,
                "angle_deg": item["angle_deg"],
                "direction_x": dx,
                "direction_y": dy,
                "lever_about_cm_m": lever,
                "k_shear_kN_m": wall_stiffness(length_m, thickness_m, True),
                "k_bending_only_kN_m": wall_stiffness(length_m, thickness_m, False),
            }
        )
    return rows


def assemble(rows: list[dict[str, Any]], cm: tuple[float, float]) -> dict[str, Any]:
    matrix = np.zeros((3, 3), dtype=float)
    for row in rows:
        vector = np.array(
            [row["direction_x"], row["direction_y"], row["lever_about_cm_m"]]
        )
        matrix += row["k_shear_kN_m"] * np.outer(vector, vector)
    coupling = matrix[:2, 2]
    shifted = -np.linalg.solve(matrix[:2, :2], coupling)
    # shifted = [delta_y, -delta_x] from the CM to the CR.
    delta_x, delta_y = -shifted[1], shifted[0]
    cr = np.array([cm[0] + delta_x, cm[1] + delta_y])
    transform = np.array(
        [[1.0, 0.0, delta_y], [0.0, 1.0, -delta_x], [0.0, 0.0, 1.0]]
    )
    matrix_at_cr = transform.T @ matrix @ transform
    return {
        "K_at_cm": matrix,
        "cr_m": cr,
        "eccentricity_m": float(np.linalg.norm(cr - np.asarray(cm))),
        "K_transformed_to_cr": matrix_at_cr,
    }


def design_response(
    rows: list[dict[str, Any]], matrix: np.ndarray
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    load = np.array([DESIGN_FORCE_KN, 0.0, 0.0])
    displacement = np.linalg.solve(matrix, load)
    sum_x = sum_y = sum_moment = 0.0
    response_rows: list[dict[str, Any]] = []
    for row in rows:
        vector = np.array(
            [row["direction_x"], row["direction_y"], row["lever_about_cm_m"]]
        )
        local_force = row["k_shear_kN_m"] * float(vector @ displacement)
        force_x = local_force * row["direction_x"]
        force_y = local_force * row["direction_y"]
        moment = local_force * row["lever_about_cm_m"]
        sum_x += force_x
        sum_y += force_y
        sum_moment += moment
        response_rows.append(
            {
                **row,
                "force_local_kN": local_force,
                "force_x_kN": force_x,
                "force_y_kN": force_y,
                "moment_about_cm_kNm": moment,
            }
        )
    return response_rows, {
        "load_at_cm": load,
        "displacement_at_cm": displacement,
        "sum_force_x_kN": sum_x,
        "sum_force_y_kN": sum_y,
        "sum_moment_about_cm_kNm": sum_moment,
    }


def write_csv(rows: list[dict[str, Any]]) -> None:
    fields = [
        "id",
        "dataset_wall_id",
        "x_m",
        "y_m",
        "length_m",
        "thickness_m",
        "angle_deg",
        "direction_x",
        "direction_y",
        "lever_about_cm_m",
        "k_shear_kN_m",
        "k_bending_only_kN_m",
        "force_local_kN",
        "force_x_kN",
        "force_y_kN",
        "moment_about_cm_kNm",
    ]
    with CSV_PATH.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: clean(row[field]) for field in fields})


def draw(reference: dict[str, Any], mass: dict[str, Any], structural: dict[str, Any]) -> None:
    fig, axis = plt.subplots(figsize=(12.8, 8.4), constrained_layout=True)
    axis.add_patch(
        Polygon(
            reference["floor"]["polygon_m"],
            closed=True,
            facecolor="#eef4fa",
            edgecolor="#263238",
            lw=1.4,
        )
    )
    for item in reference["wall_rectangles"]:
        x, y = item["centre_m"]
        half = item["size_mm"][1] / 2000.0
        angle = math.radians(item["angle_deg"])
        dx, dy = math.cos(angle), math.sin(angle)
        axis.plot(
            [x - half * dx, x + half * dx],
            [y - half * dy, y + half * dy],
            color="#263238",
            lw=3.5,
            solid_capstyle="butt",
        )
    cm = mass["cm_m"]
    cr = structural["cr_m"]
    axis.scatter([cm[0]], [cm[1]], marker="*", s=180, color="#1565c0", label="CM reference")
    axis.scatter([cr[0]], [cr[1]], marker="X", s=125, color="#c62828", label="CR reference")
    axis.plot([cm[0], cr[0]], [cm[1], cr[1]], color="#c62828", ls=":", lw=1.4)
    axis.set_title("MLStructFP966 reference-only structural benchmark (not pipeline output)")
    axis.set_xlabel("dataset x (m)")
    axis.set_ylabel("dataset y (m)")
    axis.set_aspect("equal")
    axis.grid(alpha=0.2)
    axis.legend()
    fig.savefig(FIGURE_PATH, dpi=180)
    plt.close(fig)


def main() -> None:
    reference = read_json(REFERENCE_PATH)
    mass = polygon_mass_properties(reference["floor"]["polygon_m"])
    cm = tuple(mass["cm_m"])
    rows = wall_rows(reference, cm)
    structural = assemble(rows, cm)
    response_rows, response = design_response(rows, structural["K_at_cm"])
    output = {
        "plan": "MLStructFP966",
        "basis": "independent public dataset reference; not a promoted extraction",
        "pipeline_structural_result": {
            "available": False,
            "cm_m": None,
            "cr_m": None,
            "eccentricity_m": None,
            "reason": (
                "the locked label-supervision route claimed zero elements, both VLMs "
                "resolved zero flags, and no numeric sizes were promoted"
            ),
        },
        "reference_benchmark": {
            "slab_source": "frozen MLStructFP slab polygon 972",
            "wall_source": "28 frozen MLStructFP wall rectangles treated independently",
            "mass": mass,
            "mechanics": {
                "E_kN_m2": E_KN_M2,
                "nu": POISSON,
                "G_kN_m2": G_KN_M2,
                "height_m": STOREY_HEIGHT_M,
                "shear_kappa": SHEAR_KAPPA,
                "wall_formula": "1 / (H^3/(3EI) + H/(G*(5/6)*t*L))",
                "arbitrary_angle_vector": "[cos(a), sin(a), -cos(a)*(y-ym)+sin(a)*(x-xm)]",
            },
            "wall_rectangle_count": len(rows),
            **structural,
            "design_1000kN_x_at_cm": response,
        },
        "vlm_structural_consequence": {
            "smolvlm2-256m": "unavailable: zero flags resolved and no promoted geometry",
            "llava-onevision-0.5b": "unavailable: zero flags resolved and no promoted geometry",
        },
        "checks": {
            "area_difference_from_frozen_reference_m2": abs(
                mass["effective_area_m2"] - reference["floor"]["area_m2"]
            ),
            "cm_difference_from_frozen_reference_m": float(
                np.linalg.norm(
                    np.asarray(mass["cm_m"])
                    - np.asarray(reference["floor"]["centroid_m"])
                )
            ),
            "K_symmetry_max_abs": float(
                np.max(np.abs(structural["K_at_cm"] - structural["K_at_cm"].T))
            ),
            "K_at_cr_translation_rotation_coupling_max_abs": float(
                np.max(np.abs(structural["K_transformed_to_cr"][:2, 2]))
            ),
            "design_force_x_residual_kN": response["sum_force_x_kN"] - DESIGN_FORCE_KN,
            "design_force_y_residual_kN": response["sum_force_y_kN"],
            "design_moment_residual_kNm": response["sum_moment_about_cm_kNm"],
        },
    }
    RESULT_PATH.write_text(
        json.dumps(clean(output), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_csv(response_rows)
    draw(reference, mass, structural)
    print(
        f"Reference CM=({cm[0]:.4f}, {cm[1]:.4f}) m; "
        f"CR=({structural['cr_m'][0]:.4f}, {structural['cr_m'][1]:.4f}) m; "
        f"e={structural['eccentricity_m']:.4f} m"
    )
    print("Pipeline CM/CR/e unavailable; reference benchmark kept separate")


if __name__ == "__main__":
    main()
