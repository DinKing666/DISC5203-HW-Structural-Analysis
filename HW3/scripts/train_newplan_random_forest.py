from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from run_classical_pipeline import frame_coefficients, morphology, read_image, write_image
from train_random_forest import (
    FEATURE_NAMES,
    SEED,
    feature_maps,
    predict_roi,
    rasterise_rule_target,
    sample_training,
)


ROOT = Path(__file__).resolve().parents[1]
PARAMS = ROOT / "config" / "classical-params-HW3.json"
GENERALISATION = ROOT / "config" / "generalization-plans-HW3.json"
RECORD = ROOT / "output" / "NewPlan" / "results" / "record-classical-label-supervised-NewPlan-HW3.json"
MODEL = ROOT / "models" / "random-forest-NewPlan-HW3.joblib"
RESULTS = ROOT / "output" / "NewPlan" / "results"
FIGURES = ROOT / "output" / "NewPlan" / "figures"
INTERMEDIATE = ROOT / "output" / "NewPlan" / "intermediate"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    locked_params = load_json(PARAMS)
    plan = load_json(GENERALISATION)["plans"]["NewPlan"]
    record = load_json(RECORD)
    frame = frame_coefficients(plan)
    image = read_image(ROOT / plan["source"])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    masks = morphology(gray, locked_params["classical"])
    target = rasterise_rule_target(gray.shape, record, frame)
    features = feature_maps(gray, masks)

    # The ROI is derived from the frozen 46 m x 18 m frame with a 2 m margin,
    # not tuned around the forest prediction.
    ax, sx, ay, sy = frame
    x0, x1 = sorted((int(round(ax - 2.0 * sx)), int(round(ax + 48.0 * sx))))
    y0, y1 = sorted((int(round(ay - 2.0 * sy)), int(round(ay + 20.0 * sy))))
    x0, x1 = max(0, x0), min(gray.shape[1], x1)
    y0, y1 = max(0, y0), min(gray.shape[0], y1)
    roi = np.zeros(gray.shape, dtype=bool)
    roi[y0:y1, x0:x1] = True

    x, y = sample_training(features, target, roi)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=SEED, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=80,
        max_depth=14,
        min_samples_leaf=3,
        max_features="sqrt",
        class_weight="balanced",
        oob_score=True,
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    prediction_test = model.predict(x_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, prediction_test, average="binary", zero_division=0
    )
    prediction = predict_roi(model, features, roi)
    intersection = int(np.count_nonzero((prediction == 1) & (target == 1) & roi))
    union = int(np.count_nonzero(((prediction == 1) | (target == 1)) & roi))
    iou = intersection / max(1, union)

    MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL, compress=3)
    write_image(INTERMEDIATE / "forest-rule-target-NewPlan-HW3.png", target * 255)
    write_image(INTERMEDIATE / "forest-prediction-NewPlan-HW3.png", prediction * 255)

    importance = {
        name: round(float(value), 8)
        for name, value in zip(FEATURE_NAMES, model.feature_importances_)
    }
    summary = {
        "plan": "NewPlan",
        "purpose": "kept-but-never-chosen diagnostic model trained on classical pseudo-labels",
        "selected": False,
        "selection_prohibited_reason": "The forest labels are copied from the rule-derived record, so the model cannot independently correct the rules' mistakes.",
        "locked_to_floorplan1_features_and_hyperparameters": True,
        "model_path": "models/random-forest-NewPlan-HW3.joblib",
        "random_seed": SEED,
        "roi_px": [x0, y0, x1, y1],
        "training_samples": int(y_train.size),
        "test_samples": int(y_test.size),
        "positive_fraction": round(float(y.mean()), 6),
        "oob_score": round(float(model.oob_score_), 6),
        "test_accuracy": round(float(accuracy_score(y_test, prediction_test)), 6),
        "test_precision": round(float(precision), 6),
        "test_recall": round(float(recall), 6),
        "test_f1": round(float(f1), 6),
        "rule_target_iou": round(float(iou), 6),
        "confusion_matrix": confusion_matrix(y_test, prediction_test).tolist(),
        "feature_importance": importance,
    }
    dump_json(RESULTS / "random-forest-summary-NewPlan-HW3.json", summary)
    with (RESULTS / "random-forest-feature-importance-NewPlan-HW3.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["feature", "importance"])
        writer.writerows(importance.items())

    fig, axes = plt.subplots(1, 3, figsize=(15, 6), constrained_layout=True)
    axes[0].imshow(target, cmap="gray")
    axes[0].set_title("Classical rule pseudo-label")
    axes[1].imshow(prediction, cmap="gray")
    axes[1].set_title(f"Forest copy (IoU={iou:.3f})")
    order = np.argsort(model.feature_importances_)
    axes[2].barh(
        np.asarray(FEATURE_NAMES)[order],
        model.feature_importances_[order],
        color="#3d5a80",
    )
    axes[2].set_xlabel("Gini feature importance")
    axes[2].set_title("Same locked hand features")
    for axis in axes[:2]:
        axis.axis("off")
    fig.suptitle(
        "NewPlan random forest retained as a witness, never selected",
        fontweight="bold",
    )
    fig.savefig(FIGURES / "hw3_12_newplan_random_forest_never_chosen.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
