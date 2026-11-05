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

from run_classical_pipeline import frame_coefficients, metric_to_pixel, morphology, read_image, write_image


ROOT = Path(__file__).resolve().parents[1]
PARAMS = ROOT / "config" / "classical-params-HW3.json"
RECORD = ROOT / "output" / "FloorPlan1" / "results" / "record-classical-label-supervised-HW3.json"
MODEL = ROOT / "models" / "random-forest-FloorPlan1-HW3.joblib"
RESULTS = ROOT / "output" / "FloorPlan1" / "results"
FIGURES = ROOT / "output" / "FloorPlan1" / "figures"
INTERMEDIATE = ROOT / "output" / "FloorPlan1" / "intermediate"
SEED = 21345221
FEATURE_NAMES = ["grey", "edge_strength", "ink_density_5", "ink_density_11", "ink_density_21", "ink_density_45", "stroke_45deg_density_21"]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def rasterise_rule_target(shape: tuple[int, int], record: dict, frame: tuple[float, float, float, float]) -> np.ndarray:
    target = np.zeros(shape, dtype=np.uint8)
    _, sx, _, sy = frame
    for element in record["elements"]:
        cx, cy = metric_to_pixel(*element["centre_m"], frame)
        if element["class"] == "column":
            width = element["size_mm"][0] / 1000 * abs(sx)
            height = element["size_mm"][1] / 1000 * abs(sy)
        elif element["angle_deg"] == 0:
            width = element["size_mm"][1] / 1000 * abs(sx)
            height = element["size_mm"][0] / 1000 * abs(sy)
        else:
            width = element["size_mm"][0] / 1000 * abs(sx)
            height = element["size_mm"][1] / 1000 * abs(sy)
        x0, x1 = int(round(cx - width / 2)), int(round(cx + width / 2))
        y0, y1 = int(round(cy - height / 2)), int(round(cy + height / 2))
        cv2.rectangle(target, (x0, y0), (x1, y1), 1, thickness=-1)
    return target


def feature_maps(gray: np.ndarray, masks: dict[str, np.ndarray]) -> list[np.ndarray]:
    grey = gray.astype(np.float32) / 255.0
    gx = cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3)
    edge = cv2.magnitude(gx, gy)
    edge /= max(1e-9, float(edge.max()))
    ink = (masks["ink"] > 0).astype(np.float32)
    density = [cv2.boxFilter(ink, cv2.CV_32F, (window, window), normalize=True) for window in (5, 11, 21, 45)]
    stroke_raw = (masks["diag"] > 0).astype(np.float32)
    stroke = cv2.boxFilter(stroke_raw, cv2.CV_32F, (21, 21), normalize=True)
    return [grey, edge, *density, stroke]


def sample_training(features: list[np.ndarray], target: np.ndarray, roi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    positive = np.flatnonzero((target == 1) & roi)
    negative = np.flatnonzero((target == 0) & roi)
    count = min(60000, positive.size, negative.size)
    chosen = np.r_[rng.choice(positive, count, replace=False), rng.choice(negative, count, replace=False)]
    rng.shuffle(chosen)
    x = np.column_stack([feature.ravel()[chosen] for feature in features]).astype(np.float32)
    y = target.ravel()[chosen]
    return x, y


def predict_roi(model: RandomForestClassifier, features: list[np.ndarray], roi: np.ndarray) -> np.ndarray:
    indices = np.flatnonzero(roi)
    result = np.zeros(roi.shape, dtype=np.uint8)
    flat = result.ravel()
    chunk = 200000
    for start in range(0, indices.size, chunk):
        batch = indices[start:start + chunk]
        matrix = np.column_stack([feature.ravel()[batch] for feature in features]).astype(np.float32)
        flat[batch] = model.predict(matrix).astype(np.uint8)
    return result


def main() -> None:
    params = load_json(PARAMS)
    record = load_json(RECORD)
    frame = frame_coefficients(params)
    image = read_image(ROOT / params["source"])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    masks = morphology(gray, params["classical"])
    target = rasterise_rule_target(gray.shape, record, frame)
    features = feature_maps(gray, masks)

    roi = np.zeros(gray.shape, dtype=bool)
    roi[430:2660, 600:2160] = True
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
    write_image(INTERMEDIATE / "forest-rule-target-HW3.png", target * 255)
    write_image(INTERMEDIATE / "forest-prediction-HW3.png", prediction * 255)

    importance = {name: round(float(value), 8) for name, value in zip(FEATURE_NAMES, model.feature_importances_)}
    summary = {
        "plan": "FloorPlan1",
        "purpose": "kept-but-never-chosen diagnostic model trained on classical pseudo-labels",
        "selected": False,
        "selection_prohibited_reason": "The forest labels are copied from the rule-derived record, so the model cannot independently correct the rules' mistakes.",
        "model_path": "models/random-forest-FloorPlan1-HW3.joblib",
        "random_seed": SEED,
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
    dump_json(RESULTS / "random-forest-summary-HW3.json", summary)
    with (RESULTS / "random-forest-feature-importance-HW3.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["feature", "importance"])
        writer.writerows(importance.items())

    fig, axes = plt.subplots(1, 3, figsize=(15, 6), constrained_layout=True)
    axes[0].imshow(target, cmap="gray")
    axes[0].set_title("Classical rule pseudo-label")
    axes[1].imshow(prediction, cmap="gray")
    axes[1].set_title(f"Forest copy (IoU={iou:.3f})")
    order = np.argsort(model.feature_importances_)
    axes[2].barh(np.asarray(FEATURE_NAMES)[order], model.feature_importances_[order], color="#3d5a80")
    axes[2].set_xlabel("Gini feature importance")
    axes[2].set_title("It re-learns the hand features")
    for ax in axes[:2]:
        ax.axis("off")
    fig.suptitle("Random forest retained as a witness, never selected", fontweight="bold")
    fig.savefig(FIGURES / "hw3_05_random_forest_never_chosen.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
