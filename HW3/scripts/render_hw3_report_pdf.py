from __future__ import annotations

import csv
import json
from pathlib import Path

from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    LongTable,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/pdf/21345221-report-HW3.pdf"
rl_config.invariant = 1


def load(path: str):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def rows(path: str):
    with (ROOT / path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def fonts() -> None:
    regular = next(
        path
        for path in [
            Path("C:/Windows/Fonts/arial.ttf"),
            Path("C:/Windows/Fonts/calibri.ttf"),
        ]
        if path.exists()
    )
    bold = next(
        path
        for path in [
            Path("C:/Windows/Fonts/arialbd.ttf"),
            Path("C:/Windows/Fonts/calibrib.ttf"),
        ]
        if path.exists()
    )
    pdfmetrics.registerFont(TTFont("Body", str(regular)))
    pdfmetrics.registerFont(TTFont("BodyBold", str(bold)))


def header_footer(canvas, doc) -> None:
    canvas.saveState()
    width, height = doc.pagesize
    canvas.setStrokeColor(colors.HexColor("#B8C9D3"))
    canvas.line(15 * mm, height - 14 * mm, width - 15 * mm, height - 14 * mm)
    canvas.setFont("Body", 8)
    canvas.setFillColor(colors.HexColor("#35556B"))
    canvas.drawString(15 * mm, height - 10 * mm, "DISC 5203 | HW3 | Shuhan He | 21345221")
    canvas.drawRightString(width - 15 * mm, 9 * mm, f"Page {doc.page}")
    canvas.restoreState()


def styled_table(data, widths, size=7.2, repeat=1):
    converted = []
    for row in data:
        converted.append(
            [
                cell
                if hasattr(cell, "wrap")
                else Paragraph(str(cell), STYLES["table"])
                for cell in row
            ]
        )
    table = LongTable(converted, colWidths=widths, repeatRows=repeat, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), "Body"),
                ("FONTSIZE", (0, 0), (-1, -1), size),
                ("LEADING", (0, 0), (-1, -1), size + 2),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#AFC1CC")),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#204A6B")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "BodyBold"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F6F8")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 2.5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2.5),
                ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
            ]
        )
    )
    return table


def fit_image(path: Path, width_mm: float, height_mm: float):
    image = Image(str(path))
    ratio = min(width_mm * mm / image.imageWidth, height_mm * mm / image.imageHeight)
    image.drawWidth = image.imageWidth * ratio
    image.drawHeight = image.imageHeight * ratio
    return image


def matrix_text(values) -> str:
    return "\n".join(
        "[ " + "  ".join(f"{value: .4e}" for value in row) + " ]"
        for row in values
    )


def model_rows(plan, comparison, verdict):
    output = []
    for model in comparison["models"]:
        output.append(
            [
                plan,
                model["model_key"],
                model["flags_total"],
                model["flags_resolved"],
                model["flags_broken"],
                model["flags_unresolved"],
                model["mean_reported_confidence"] or "n/a",
                model["mean_size_error_mm"]
                if model["mean_size_error_mm"] is not None
                else "n/a",
                verdict if model["model_key"] in verdict else "witness",
            ]
        )
    return output


def main() -> None:
    fonts()
    global STYLES
    base = getSampleStyleSheet()
    STYLES = {
        "title": ParagraphStyle(
            "title", parent=base["Title"], fontName="BodyBold", fontSize=21,
            leading=25, textColor=colors.HexColor("#173F5F"), alignment=TA_LEFT,
            spaceAfter=5 * mm,
        ),
        "sub": ParagraphStyle(
            "sub", parent=base["Normal"], fontName="Body", fontSize=10.5,
            leading=15, textColor=colors.HexColor("#46697A"), spaceAfter=2 * mm,
        ),
        "h1": ParagraphStyle(
            "h1", parent=base["Heading1"], fontName="BodyBold", fontSize=14,
            leading=17, textColor=colors.HexColor("#173F5F"), spaceBefore=3 * mm,
            spaceAfter=2 * mm, keepWithNext=1,
        ),
        "h2": ParagraphStyle(
            "h2", parent=base["Heading2"], fontName="BodyBold", fontSize=10.5,
            leading=13, textColor=colors.HexColor("#35556B"), spaceBefore=2 * mm,
            spaceAfter=1.5 * mm, keepWithNext=1,
        ),
        "body": ParagraphStyle(
            "body", parent=base["BodyText"], fontName="Body", fontSize=8.8,
            leading=12.4, textColor=colors.HexColor("#202B33"), spaceAfter=2 * mm,
        ),
        "table": ParagraphStyle(
            "table", parent=base["BodyText"], fontName="Body", fontSize=7,
            leading=9, textColor=colors.HexColor("#202B33"),
        ),
        "cap": ParagraphStyle(
            "cap", parent=base["BodyText"], fontName="Body", fontSize=7.3,
            leading=9, textColor=colors.HexColor("#5B6970"), alignment=TA_CENTER,
            spaceAfter=2.5 * mm,
        ),
        "call": ParagraphStyle(
            "call", parent=base["BodyText"], fontName="BodyBold", fontSize=9.4,
            leading=14, textColor=colors.white, backColor=colors.HexColor("#204A6B"),
            borderPadding=7, spaceAfter=3 * mm,
        ),
        "code": ParagraphStyle(
            "code", parent=base["Code"], fontName="Courier", fontSize=6.8,
            leading=9, textColor=colors.HexColor("#243746"), backColor=colors.HexColor("#F2F6F8"),
            borderPadding=5, spaceAfter=2 * mm,
        ),
    }
    P = lambda text, style="body": Paragraph(text, STYLES[style])

    classical = load("output/FloorPlan1/results/score-sheet-classical-HW3.json")
    record = load("output/FloorPlan1/results/record-classical-label-supervised-HW3.json")
    forest = load("output/FloorPlan1/results/random-forest-summary-HW3.json")
    flags = load("output/FloorPlan1/results/flags-manifest-HW3.json")
    vlm = load("output/FloorPlan1/results/vlm-comparison-FloorPlan1-HW3.json")
    hand = load("output/FloorPlan1/results/hand-structural-FloorPlan1-HW3.json")
    ops = load("output/FloorPlan1/results/opensees-FloorPlan1-HW3.json")
    modified = load("output/FloorPlan1/results/modified-hand-FloorPlan1-HW3.json")
    modified_ops = load("output/FloorPlan1/results/opensees-modified-FloorPlan1-HW3.json")
    wall_rows = rows("output/FloorPlan1/results/wall-stiffness-and-forces-FloorPlan1-HW3.csv")
    new_score = load("output/NewPlan/results/score-sheet-classical-NewPlan-HW3.json")
    new_forest = load("output/NewPlan/results/random-forest-summary-NewPlan-HW3.json")
    new_vlm = load("output/NewPlan/results/vlm-comparison-NewPlan-HW3.json")
    new_hand = load("output/NewPlan/results/hand-structural-NewPlan-HW3.json")
    found_score = load("output/MLStructFP966/results/score-sheet-classical-MLStructFP966-HW3.json")
    found_forest = load("output/MLStructFP966/results/random-forest-summary-MLStructFP966-HW3.json")
    found_vlm = load("output/MLStructFP966/results/vlm-comparison-MLStructFP966-HW3.json")
    found_struct = load("output/MLStructFP966/results/reference-structural-benchmark-MLStructFP966-HW3.json")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = BaseDocTemplate(
        str(OUT), pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm,
        topMargin=19 * mm, bottomMargin=15 * mm, title="DISC 5203 Homework 3",
        author="Shuhan He",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="main")
    doc.addPageTemplates([PageTemplate(id="all", frames=frame, onPage=header_footer)])

    story = []
    logo = ROOT / "assets/UST_logo.png"
    if logo.exists():
        story.append(fit_image(logo, 17, 17))
    story += [
        Spacer(1, 8 * mm),
        P("Floor-Plan Reading and Rigid-Diaphragm Centre of Stiffness", "title"),
        P("Classical label supervision, two VLM readers, hand stiffness and an OpenSees check", "sub"),
        Spacer(1, 3 * mm),
        styled_table(
            [
                ["Course", "Computer (& AI) Methods for Structural Engineering (DISC 5203)"],
                ["Student", "Shuhan He / 何抒翰"],
                ["Student ID", "21345221"],
                ["Assigned plan", "Floor Plan 1 (final digit 1)"],
                ["Prepared", "7 November 2026"],
                ["Repository", "Cumulative HW1 + HW2 + HW3 repository"],
            ],
            [35 * mm, 120 * mm], 8.7, 0,
        ),
        Spacer(1, 7 * mm),
        P(
            "18 elements | LLaVA 3/4 flags | CM (5.4605, 8.7915) m | "
            "CR (5.5488, 9.8946) m | OpenSees agreement 9.33e-10 m", "call"
        ),
        P("Executive result", "h1"),
        P(
            "The required L09 order is preserved: fixed classical rules, one printed label "
            "per element, a retained but never-selected forest, then two VLMs restricted to "
            "frozen flag crops. LLaVA clears T4, T10 and T11; T3 remains unresolved and is "
            "excluded from the primary lateral model. The resulting eccentricity is 1.10660 m."
        ),
        P(
            "Rebuilding T3 at 309 mm and thickening T7 to 365 mm, without moving the mass "
            "layout or either centreline, reduces eccentricity to 0.000268 m (99.9758%). "
            "The same locked reader succeeds on NewPlan and fails safely on public MLStructFP966."
        ),
        PageBreak(),
        P("1. Fixed pipeline and audit boundary", "h1"),
        P(
            "Reader: grey&lt;225; 11-px 45-degree opening; 17/21-px orthogonal openings; "
            "4:1 shape rule; OCR twice at threshold 0.85. Pixels place; printed text sizes. "
            "VLMs receive local crops only and may not alter coordinates."
        ),
        fit_image(ROOT / "output/FloorPlan1/figures/hw3_01_classical_morphology.png", 176, 188),
        P("Figure 1. FloorPlan1 fixed threshold and morphology evidence.", "cap"),
        P(
            f"The pass keeps {classical['connected_components_kept']} components and "
            f"{classical['shape_rule_elongated_candidates']} elongated candidates. Both OCR "
            f"passes recover all 18 names; label supervision closes {record['floor']['effective_area_m2']:.4f} m2 "
            f"and {record['checks']['sum_axial_kN']:.3f} kN on five storeys."
        ),
        PageBreak(),
        P("2. Label-supervised record and frozen flags", "h1"),
        fit_image(ROOT / "output/FloorPlan1/figures/hw3_04_label_supervised_record.png", 170, 177),
        P("Figure 2. Immutable 7-column + 11-wall record; pixels place and text sizes.", "cap"),
    ]
    flag_table = [["Label", "Classical flag", "Expected size", "Crop hash"]]
    for item in flags["flags"]:
        flag_table.append(
            [item["label"], item["flag"], item["answer_schema"]["size_mm"], item["crop_sha256"][:12]]
        )
    story += [
        styled_table(flag_table, [14 * mm, 71 * mm, 29 * mm, 38 * mm], 7.2),
        P(
            "The original JSON is never overwritten. Every crop, question, order, reference "
            "and confidence threshold is frozen before either model runs."
        ),
        PageBreak(),
        P("3. Kept-but-never-chosen random forests", "h1"),
        P(
            "Seven per-pixel features, 80 trees, depth 14, minimum leaf 3, balanced classes "
            "and seed 21345221 are identical across plans. Pseudo-labels come from the rules, "
            "so forest selection is prohibited."
        ),
        styled_table(
            [
                ["Plan", "OOB", "Test F1", "Rule IoU", "Main evidence", "Selected"],
                ["FloorPlan1", forest["oob_score"], forest["test_f1"], forest["rule_target_iou"], "21/45 ink + 45deg", False],
                ["NewPlan", new_forest["oob_score"], new_forest["test_f1"], new_forest["rule_target_iou"], "21/45 ink + 45deg", False],
                ["MLStructFP966", found_forest["oob_score"], found_forest["test_f1"], found_forest["rule_target_iou"], "45-px ink", False],
            ],
            [29 * mm, 20 * mm, 21 * mm, 22 * mm, 42 * mm, 21 * mm], 7.2,
        ),
        fit_image(ROOT / "output/FloorPlan1/figures/hw3_05_random_forest_never_chosen.png", 178, 92),
        P("Figure 3. High held-out F1 does not create an independent correction route.", "cap"),
        PageBreak(),
        P("4. Two VLMs, one gate", "h1"),
        P(
            "Models: SmolVLM2-256M revision 067788... and LLaVA-OneVision-Qwen2-0.5B "
            "revision 74dd0b.... CPU float32, greedy decoding, seed 21345221. Required keys: "
            "label, size_mm, continuous, confidence, reason. A valid answer must exceed 0.85, "
            "agree with the independent reference and not contradict its own reason."
        ),
    ]
    perf = [["Plan", "Model", "N", "Resolved", "Broken", "Unresolved", "Conf", "Err mm", "Verdict"]]
    perf += model_rows("FloorPlan1", vlm, "llava-onevision-0.5b promoted")
    perf += model_rows("NewPlan", new_vlm, "llava-onevision-0.5b promoted")
    perf += model_rows("MLStructFP966", found_vlm, "no model")
    story += [
        styled_table(perf, [24 * mm, 31 * mm, 10 * mm, 17 * mm, 15 * mm, 18 * mm, 15 * mm, 16 * mm, 28 * mm], 6.4),
        P(
            "FloorPlan1: LLaVA clears T4/T10/T11, but T3 remains contradictory. Both actual "
            "VLM structural routes therefore sit 0.91416 m from the independent-reference CR. "
            "NewPlan: LLaVA clears its one OCR audit. MLStructFP966: both models resolve zero; "
            "the zero-resolved tie is rejected and no structural record is promoted."
        ),
        P(
            "Official model cards: https://huggingface.co/HuggingFaceTB/SmolVLM2-256M-Video-Instruct "
            "and https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf.", "cap"
        ),
        PageBreak(),
        P("5. Hand CM and 3x3 stiffness", "h1"),
        P(
            "Wall stiffness includes bending and shear: k = 1/[H3/(3EI)+H/(G Av)], "
            "I=tL3/12 and Av=(5/6)tL. Signed slab regions give area 180.1285 m2, "
            "CM (5.460451, 8.791522) m, 160.665 t/storey and 803.325 t on five storeys."
        ),
        styled_table(
            [["Region", "Area m2", "cx", "cy"]]
            + [
                [item["name"], f'{item["area_m2"]:.4f}', f'{item["centroid_x_m"]:.3f}', f'{item["centroid_y_m"]:.3f}']
                for item in hand["mass"]["regions"]
            ],
            [65 * mm, 29 * mm, 29 * mm, 29 * mm], 7.4,
        ),
        P("Primary hand matrices", "h2"),
        Preformatted("K\n" + matrix_text(hand["promoted_shear_included"]["K_at_cm"]) + "\n\nF=K^-1\n" + matrix_text(hand["promoted_shear_included"]["F_at_cm"]), STYLES["code"]),
        P(
            "CR = (5.548845, 9.894589) m and e = 1.106603 m. Bending-only e = 2.012353 m; "
            "the shear term is material and retained."
        ),
        PageBreak(),
        P("6. Wall stiffness and 1000 kN push + turn", "h1"),
    ]
    wall_table = [["Wall", "Dir", "x", "y", "L", "t", "Active", "k shear", "Push", "Turn", "Total"]]
    for row in wall_rows:
        wall_table.append(
            [
                row["id"], row["direction"], row["x_m"], row["y_m"], row["length_m"],
                row["thickness_m"], row["continuous"], f'{float(row["k_shear_kN_m"]):.0f}',
                f'{float(row["force_push_kN"]):.1f}', f'{float(row["force_turn_kN"]):.1f}',
                f'{float(row["force_total_kN"]):.1f}',
            ]
        )
    story += [
        styled_table(wall_table, [13 * mm, 11 * mm, 16 * mm, 16 * mm, 13 * mm, 13 * mm, 17 * mm, 23 * mm, 18 * mm, 18 * mm, 18 * mm], 6.5),
        P(
            "The +x load at CM equals a 1000 kN push through CR plus 1103.067 kN m torque. "
            "Forces close to 1000 kN in x, zero in cross-wall y and the applied torque."
        ),
        fit_image(ROOT / "output/FloorPlan1/figures/hw3_06_cm_cr_and_wall_forces.png", 176, 105),
        P("Figure 4. Main-plan CM/CR and wall-force decomposition.", "cap"),
        PageBreak(),
        P("7. OpenSees rigid-diaphragm check", "h1"),
        P(
            "One 3D ElasticTimoshenkoBeam per active wall; strong-axis I=tL3/12 and "
            "Av=(5/6)tL. Wall tops use rigidDiaphragm(3, master, *slaves) with Transformation "
            "constraints. Three unit loads at CM recover F and K."
        ),
        Preformatted("OpenSees F\n" + matrix_text(ops["flexibility_at_cm"]) + "\n\nOpenSees K\n" + matrix_text(ops["stiffness_at_cm"]), STYLES["code"]),
        styled_table(
            [
                ["Check", "Result"],
                ["CR difference from hand", f"{ops['comparison_to_hand']['cr_difference_norm_m']:.3e} m"],
                ["Maximum relative K difference", f"{ops['comparison_to_hand']['max_abs_K_relative_difference']:.3e}"],
                ["Rotation for unit push through CR", f"{ops['comparison_to_hand']['unit_push_through_cr_rotation_rad']} rad"],
                ["1000 kN wall sum", f"{ops['comparison_to_hand']['design_wall_sum_x_kN']:.6f} kN"],
                ["Cross-wall sum", f"{ops['comparison_to_hand']['design_cross_wall_sum_y_kN']:.3e} kN"],
                ["Push + turn max residual", f"{ops['comparison_to_hand']['max_wall_force_additivity_residual_kN']:.1e} kN"],
            ],
            [67 * mm, 88 * mm], 7.5,
        ),
        P(
            "Documentation: https://openseespydoc.readthedocs.io/en/latest/src/ElasticTimoshenkoBeam.html; "
            "https://openseespydoc.readthedocs.io/en/latest/src/rigidDiaphragm.html; "
            "https://openseespydoc.readthedocs.io/en/latest/src/TransformationMethod.html.", "cap"
        ),
        PageBreak(),
        P("8. Make CM and CR coincide", "h1"),
        P(
            "Mass, openings, columns, grid and all wall centrelines remain fixed. T3 is rebuilt "
            "continuous at 309 mm; T7 is thickened from 250 to 365 mm."
        ),
        styled_table(
            [
                ["Quantity", "Before", "After"],
                ["CR x m", f"{modified['before_after']['cr_before_m'][0]:.6f}", f"{modified['before_after']['cr_after_m'][0]:.6f}"],
                ["CR y m", f"{modified['before_after']['cr_before_m'][1]:.6f}", f"{modified['before_after']['cr_after_m'][1]:.6f}"],
                ["Eccentricity m", f"{modified['before_after']['eccentricity_before_m']:.6f}", f"{modified['before_after']['eccentricity_after_m']:.6f}"],
                ["1000 kN torque kN m", "1103.067", "0.218"],
                ["Peak wall force kN", f"{modified['before_after']['peak_abs_wall_force_before_kN']:.3f}", f"{modified['before_after']['peak_abs_wall_force_after_kN']:.3f}"],
                ["Ttheta/Tx", f"{modified['before_after']['period_ratio_Ttheta_over_Tx_before']:.4f}", f"{modified['before_after']['period_ratio_Ttheta_over_Tx_after']:.4f}"],
            ],
            [65 * mm, 45 * mm, 45 * mm], 7.7,
        ),
        fit_image(ROOT / "output/FloorPlan1/figures/hw3_07_modification_before_after.png", 178, 145),
        P("Figure 5. Same mass/grid; stiffness is retuned at two existing centrelines.", "cap"),
        P(
            f"Eccentricity reduction = {modified['before_after']['eccentricity_reduction_percent']:.4f}%. "
            f"Modified OpenSees CR difference = {modified_ops['comparison_to_hand']['cr_difference_norm_m']:.2e} m."
        ),
        PageBreak(),
        P("9. Why coincidence matters", "h1"),
        P(
            "Before modification, e=1.10660 m creates 1103.07 kN m torque. The torsional "
            "radius is sqrt(29.758)=5.455 m; using 1 + e d/r_t^2, T2 is amplified to 1.3260 "
            "times its no-turn share. Peak absolute force falls 604.08 -> 582.19 kN."
        ),
        P(
            "Ttheta/Tx falls 1.1068 -> 1.0266. The pre-change first mode is 68.98% torsional; "
            "after decoupling it is nearly pure torsion rather than mixed translation-torsion. "
            "Coincidence removes coupling but need not reorder modes."
        ),
        P(
            "CM=CR removes torque for one uniform floor under a uniform lateral action. It does "
            "not remove torsion from height-varying layouts, nonuniform ground motion, cracking, "
            "uncertainty or higher modes. Extraction matters: the unresolved T3 shifts the VLM "
            "route 0.914 m from the independent-reference CR."
        ),
        P("Key checks", "h2"),
        styled_table(
            [
                ["Check", "Before", "After"],
                ["Unit push through CR rotation", "0", "0"],
                ["Push through CM rotation", "nonzero", "near zero after e reduction"],
                ["Push + turn superposition", "exact to saved precision", "exact to saved precision"],
                ["Mass layout changed", "no", "no"],
            ],
            [66 * mm, 44 * mm, 45 * mm], 7.6,
        ),
        PageBreak(),
        P("10. Course holdout: NewPlan", "h1"),
        P(
            f"Unchanged parameters recover {new_score['named_elements_found']}/38 names: 30 columns, "
            "8 walls. T1 is reconstructed from split OCR and is the only flag. LLaVA resolves "
            "1/1 at 0.9; Smol resolves 0/1. Area closes 828.0 m2 and five-storey axial load "
            "closes 36,225 kN."
        ),
        fit_image(ROOT / "output/NewPlan/figures/hw3_11_newplan_label_supervised_record.png", 177, 167),
        P("Figure 6. NewPlan 38-element label-supervised record.", "cap"),
        P(
            f"CM=(23.0000, 9.0000) m; CR=({new_hand['promoted_shear_included']['cr_m'][0]:.5f}, "
            f"{new_hand['promoted_shear_included']['cr_m'][1]:.5f}) m; "
            f"e={new_hand['promoted_shear_included']['eccentricity_m']:.5f} m."
        ),
        PageBreak(),
        P("11. NewPlan structural result", "h1"),
        fit_image(ROOT / "output/NewPlan/figures/hw3_13_newplan_cm_cr_and_wall_forces.png", 178, 180),
        P("Figure 7. Extracted and frozen-reference CR differ by 28.8 mm.", "cap"),
        P(
            "T1's classical continuity was already true; its flag concerned split OCR. Both VLM "
            "structural states therefore retain eight walls. The VLM benefit is independent audit "
            "certainty, not coordinate or stiffness change."
        ),
        PageBreak(),
        P("12. Public holdout: MLStructFP Floor 966", "h1"),
        P(
            "Source: public MLStructFP test Floor 966 (MIT), 9600x6784 transparent blue-line "
            "CAD, irregular slab, 20 wall groups/28 rectangles, one 8.5-degree wall and zero "
            "printed C/T size strings. https://github.com/MLSTRUCT/MLStructFP; "
            "https://doi.org/10.1016/j.autcon.2023.105132."
        ),
        fit_image(ROOT / "output/MLStructFP966/figures/hw3_16_mlstruct_label_supervision_failure.png", 178, 145),
        P("Figure 8. Fixed morphology: 15 unclaimed candidates, but zero auditable elements.", "cap"),
        styled_table(
            [
                ["Metric", "Result"],
                ["Connected components", found_score["connected_components_kept"]],
                ["Elongated candidates", found_score["shape_rule_elongated_candidates"]],
                ["Accepted OCR labels", found_score["ocr_above_threshold"]],
                ["Claimed elements", found_score["label_claimed_elements"]],
                ["Unique public rectangles hit", "14/28 = 50%"],
                ["VLM result", "both 0/15; no promoted reader"],
            ],
            [84 * mm, 71 * mm], 7.6,
        ),
        PageBreak(),
        P("13. Public-plan forest and safe failure", "h1"),
        fit_image(ROOT / "output/MLStructFP966/figures/hw3_18_mlstruct_random_forest_never_chosen.png", 178, 100),
        P("Figure 9. Forest F1 0.6743 and full-image IoU 0.2214; it copies rejected composites.", "cap"),
        P(
            "The forest has no printed IDs/sizes and is never selected. Both VLMs resolve 0/15. "
            "NOT_VISIBLE is not a wall size; therefore the accepted pipeline CM, CR and e are "
            "unavailable. The result is a safe failure rather than public-annotation leakage."
        ),
        P("Reference-only benchmark", "h2"),
        P(
            f"Using the frozen public slab and all 28 frozen rectangles only for comparison: "
            f"CM=({found_struct['reference_benchmark']['mass']['cm_m'][0]:.6f}, "
            f"{found_struct['reference_benchmark']['mass']['cm_m'][1]:.6f}) m; "
            f"CR=({found_struct['reference_benchmark']['cr_m'][0]:.6f}, "
            f"{found_struct['reference_benchmark']['cr_m'][1]:.6f}) m; "
            f"e={found_struct['reference_benchmark']['eccentricity_m']:.6f} m. "
            "These are not pipeline outputs."
        ),
        PageBreak(),
        P("14. Public reference benchmark", "h1"),
        fit_image(ROOT / "output/MLStructFP966/figures/hw3_19_mlstruct_reference_cm_cr.png", 178, 185),
        P("Figure 10. Reference-only CM/CR; title prevents confusion with extraction output.", "cap"),
        P(
            "The arbitrary-angle vector is [cos(a), sin(a), -cos(a)(y-ym)+sin(a)(x-xm)]. "
            "The transformed CR matrix has zero translation-rotation coupling to saved precision; "
            "a 1000 kN +x reference load closes force and moment equilibrium."
        ),
        PageBreak(),
        P("15. Cross-plan verdict", "h1"),
        styled_table(
            [
                ["Plan", "Elements", "Flags", "Best VLM", "Pipeline CM", "Pipeline CR", "e", "Verdict"],
                ["FloorPlan1", 18, 4, "LLaVA 3/4", "(5.4605,8.7915)", "(5.5488,9.8946)", "1.1066", "accepted"],
                ["NewPlan", 38, 1, "LLaVA 1/1", "(23.0000,9.0000)", "(23.0075,9.9459)", "0.9459", "accepted"],
                ["MLStructFP966", 0, 15, "none 0/15", "unavailable", "unavailable", "unavailable", "safe failure"],
            ],
            [25 * mm, 15 * mm, 14 * mm, 25 * mm, 31 * mm, 31 * mm, 18 * mm, 26 * mm], 6.5,
        ),
        P(
            "What generalized: frozen morphology on isolated bars, deterministic forests, crop "
            "hashes, a non-regressing VLM gate and a closing structural calculation when numeric "
            "geometry exists. What did not: the one-label/one-element premise on no-text CAD, "
            "4:1 separation of composite walls, and small-model protocol following."
        ),
        P("Reproducibility", "h1"),
        P(
            "Scripts regenerate every deterministic output; real VLM outputs are committed and "
            "routine checks use --score-only. OpenSees runs in the isolated Python 3.12 target. "
            "The full command sequence and deliverable index are in README-HW3.md and the Markdown report."
        ),
        P("Limitations", "h2"),
        P(
            "Raster OCR/morphology are style-sensitive. Structural models are linear, elastic, "
            "uncracked and single-storey; diaphragm flexibility, nonlinear cracking, soil, vertical "
            "irregularity and nonuniform excitation are excluded. Public annotations are defensible "
            "references, not truth."
        ),
        P("Deliverables", "h1"),
        P(
            "Markdown + PDF report; immutable and promoted records; both VLM outputs per plan; "
            "three retained forests; hand/OpenSees JSON and wall-force CSVs; figures 01-19; exact "
            "requirements; and the complete material iteration log."
        ),
    ]
    doc.build(story)
    print(OUT)


if __name__ == "__main__":
    main()
