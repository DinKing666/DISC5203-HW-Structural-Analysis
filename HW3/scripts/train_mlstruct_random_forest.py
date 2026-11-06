from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from run_classical_pipeline import morphology, write_image
from run_mlstruct_classical import alpha_to_white, downsample
from train_random_forest import FEATURE_NAMES, SEED, feature_maps, predict_roi, sample_training


ROOT = Path(__file__).resolve().parents[1]
PARAMS = ROOT / "config/classical-params-HW3.json"
GENERALISATION = ROOT / "config/generalization-plans-HW3.json"
RECORD = (
    ROOT
    / "output/MLStructFP966/results/record-classical-label-supervised-MLStructFP966-HW3.json"
)
MODEL = ROOT / "models/random-forest-MLStructFP966-HW3.joblib"
RESULTS = ROOT / "output/MLStructFP966/results"
FIGURES = ROOT / "output/MLStructFP966/figures"
INTERMEDIATE = ROOT / "output/MLStructFP966/intermediate"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def pseudo_label_target(shape: tuple[int, int], record: dict) -> np.ndarray:
    target = np.zeros(shape, dtype=np.uint8)
    for candidate in record["unclaimed_candidates"]:
        x, y, width, height = candidate["bbox_px"]
        cv2.rectangle(target, (x, y), (x + width - 1, y + height - 1), 1, thickness=-1)
    return target


def main() -> None:
    params = read_json(PARAMS)
    plan = read_json(GENERALISATION)["plans"]["MLStructFP966"]
    record = read_json(RECORD)
    image = alpha_to_white(ROOT / plan["source"])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    masks = morphology(gray, params["classical"])
    target = pseudo_label_target(gray.shape, record)
    features = feature_maps(gray, masks)
    roi = np.ones(gray.shape, dtype=bool)

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
    intersection = int(np.count_nonzero((prediction == 1) & (target == 1)))
    union = int(np.count_nonzero((prediction == 1) | (target == 1)))
    iou = intersection / max(1, union)

    MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL, compress=3)
    write_image(INTERMEDIATE / "forest-rule-target-MLStructFP966-HW3.png", target * 255)
    write_image(INTERMEDIATE / "forest-prediction-MLStructFP966-HW3.png", prediction * 255)
    importance = {
        name: round(float(value), 8)
        for name, value in zip(FEATURE_NAMES, model.feature_importances_)
    }
    summary = {
        "plan": "MLStructFP966",
        "purpose": "kept-but-never-chosen diagnostic model trained on 15 classical shape-rule pseudo-label boxes",
        "selected": False,
        "selection_prohibited_reason": "The forest target copies unclaimed classical candidate boxes and contains no printed label or element-size supervision; it cannot create an auditable structural record.",
        "locked_to_floorplan1_features_and_hyperparameters": True,
        "public_annotation_used_for_training": False,
        "model_path": "models/random-forest-MLStructFP966-HW3.joblib",
        "random_seed": SEED,
        "prediction_roi": "entire 9600x6784 source; no annotation-defined crop",
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
    write_json(RESULTS / "random-forest-summary-MLStructFP966-HW3.json", summary)
    with (RESULTS / "random-forest-feature-importance-MLStructFP966-HW3.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(["feature", "importance"])
        writer.writerows(importance.items())

    fig, axes = plt.subplots(1, 3, figsize=(15, 6), constrained_layout=True)
    axes[0].imshow(downsample(target), cmap="gray")
    axes[0].set_title("15 classical candidate boxes")
    axes[1].imshow(downsample(prediction), cmap="gray")
    axes[1].set_title(f"Forest copy (IoU={iou:.3f})")
    order = np.argsort(model.feature_importances_)
    axes[2].barh(
        np.asarray(FEATURE_NAMES)[order], model.feature_importances_[order], color="#3d5a80"
    )
    axes[2].set_xlabel("Gini feature importance")
    axes[2].set_title("Same locked hand features")
    for axis in axes[:2]:
        axis.axis("off")
    fig.suptitle(
        "MLStructFP966 random forest is retained but cannot supply labels or sizes",
        fontweight="bold",
    )
    fig.savefig(FIGURES / "hw3_18_mlstruct_random_forest_never_chosen.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
