"""Run two frozen local VLMs on the same FloorPlan1 flag crops.

The script deliberately keeps model generation separate from scoring.  Raw model
text is retained verbatim; only strict JSON answers can pass the independent-
reference gate.  No model is ever given the full floor plan and no answer can
change an element coordinate.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import re
import shutil
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    LlavaOnevisionForConditionalGeneration,
)


ROOT = Path(__file__).resolve().parents[2]
PLAN = "FloorPlan1"
MANIFEST_PATH = ROOT / "HW3/output/FloorPlan1/results/flags-manifest-HW3.json"
REFERENCE_PATH = ROOT / "HW3/config/reference-manual-FloorPlan1-HW3.json"
CLASSICAL_PATH = (
    ROOT / "HW3/output/FloorPlan1/results/record-classical-label-supervised-HW3.json"
)
RESULTS_DIR = ROOT / "HW3/output/FloorPlan1/results"
SEED = 21345221
MAX_NEW_TOKENS = 160
PROMPT_VERSION = 2
PARSER_VERSION = 2

MODELS = {
    "smolvlm2-256m": {
        "model_id": "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        "revision": "067788b187b95ebe7b2e040b3e4299e342e5b8fd",
        "architecture": "Idefics3 / SmolLM2",
        "loader": "smol",
        "result": "vlm-smolvlm2-256m-FloorPlan1-HW3.json",
    },
    "llava-onevision-0.5b": {
        "model_id": "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
        "revision": "74dd0bf867a4cda7950c17663794267c60cf4b40",
        "architecture": "LLaVA-OneVision SO400M / Qwen2",
        "loader": "llava",
        "result": "vlm-llava-onevision-0.5b-FloorPlan1-HW3.json",
    },
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frozen_prompt(flag: dict[str, Any]) -> str:
    return (
        f'{flag["question"]}\n'
        "Inspect the crop, then return exactly one valid JSON object with exactly these "
        "five keys: label, size_mm, continuous, confidence, reason. label and size_mm "
        "must be strings read from the image; continuous must be true or false; confidence "
        "must be your non-placeholder decimal judgment from 0 to 1; reason must be one "
        "short string. Output JSON only, with no markdown and no text or punctuation "
        "outside the closing brace. Do not merely repeat the question."
    )


def load_model(spec: dict[str, str]) -> tuple[Any, Any]:
    processor = AutoProcessor.from_pretrained(
        spec["model_id"], revision=spec["revision"]
    )
    common = {
        "revision": spec["revision"],
        "torch_dtype": torch.float32,
        "low_cpu_mem_usage": True,
    }
    if spec["loader"] == "smol":
        model = AutoModelForImageTextToText.from_pretrained(
            spec["model_id"], **common
        )
    else:
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            spec["model_id"], **common
        )
    model.eval()
    return processor, model


def generate_answer(
    processor: Any, model: Any, image: Image.Image, prompt: str
) -> tuple[str, float]:
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    formatted = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    inputs = processor(images=image, text=formatted, return_tensors="pt")
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,
            use_cache=True,
        )
    elapsed = time.perf_counter() - started
    prompt_tokens = inputs["input_ids"].shape[-1]
    raw = processor.decode(output[0][prompt_tokens:], skip_special_tokens=True).strip()
    return raw, elapsed


def parse_strict_json(
    raw: str,
) -> tuple[dict[str, Any] | None, str | None, list[str]]:
    normalizations: list[str] = []
    candidate = raw.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(\{.*\})\s*```", candidate, re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
        normalizations.append("removed one outer JSON code fence")
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return None, f"JSON parse failed after allowed syntax normalization: {exc.msg}", normalizations
    required = {"label", "size_mm", "continuous", "confidence", "reason"}
    if not isinstance(parsed, dict) or set(parsed) != required:
        return None, "answer does not contain exactly the five required keys", normalizations
    if not isinstance(parsed["label"], str) or not isinstance(parsed["size_mm"], str):
        return None, "label and size_mm must be strings", normalizations
    if isinstance(parsed["continuous"], str) and parsed["continuous"].lower() in {
        "true",
        "false",
    }:
        parsed["continuous"] = parsed["continuous"].lower() == "true"
        normalizations.append("coerced quoted JSON boolean without changing its value")
    if not isinstance(parsed["continuous"], bool):
        return None, "continuous must be a JSON boolean", normalizations
    if isinstance(parsed["confidence"], str):
        try:
            parsed["confidence"] = float(parsed["confidence"])
        except ValueError:
            pass
        else:
            normalizations.append("coerced quoted decimal without changing its value")
    if isinstance(parsed["confidence"], bool) or not isinstance(
        parsed["confidence"], (int, float)
    ):
        return None, "confidence must be numeric", normalizations
    if not 0.0 <= float(parsed["confidence"]) <= 1.0:
        return None, "confidence lies outside 0..1", normalizations
    if not isinstance(parsed["reason"], str):
        return None, "reason must be a string", normalizations
    parsed["confidence"] = float(parsed["confidence"])
    return parsed, None, normalizations


def dimensions(size: str) -> tuple[int, int] | None:
    match = re.fullmatch(r"\s*(\d{2,4})\s*[xX×]\s*(\d{2,4})\s*", size)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def size_error_mm(answer: str, expected: str) -> float | None:
    if answer.strip().upper() == "NOT_VISIBLE" and expected.strip().upper() == "NOT_VISIBLE":
        return 0.0
    if "NOT_VISIBLE" in {answer.strip().upper(), expected.strip().upper()}:
        return None
    got = dimensions(answer)
    wanted = dimensions(expected)
    if got is None or wanted is None:
        return None
    # Rectangular dimensions are order-invariant: a printed 3500x400 wall and
    # the normalised engineering record 400x3500 describe the same section.
    got_ordered = sorted(got)
    wanted_ordered = sorted(wanted)
    return (
        abs(got_ordered[0] - wanted_ordered[0])
        + abs(got_ordered[1] - wanted_ordered[1])
    ) / 2.0


def comparison_path() -> Path:
    return RESULTS_DIR / f"vlm-comparison-{PLAN}-HW3.json"


def promoted_path() -> Path:
    return RESULTS_DIR / f"record-promoted-{PLAN}-HW3.json"


def run_model(model_key: str, manifest: dict[str, Any]) -> Path:
    spec = MODELS[model_key]
    torch.manual_seed(SEED)
    processor, model = load_model(spec)
    answers: list[dict[str, Any]] = []
    total_started = time.perf_counter()
    for index, flag in enumerate(manifest["flags"], start=1):
        crop_path = ROOT / "HW3" / flag["crop"]
        actual_hash = sha256_file(crop_path)
        if actual_hash != flag["crop_sha256"]:
            raise RuntimeError(f"crop hash changed: {crop_path}")
        prompt = frozen_prompt(flag)
        with Image.open(crop_path) as source:
            image = source.convert("RGB")
        raw, elapsed = generate_answer(processor, model, image, prompt)
        parsed, parse_error, normalizations = parse_strict_json(raw)
        answers.append(
            {
                "order": index,
                "flag_id": flag["flag_id"],
                "label_expected": flag["label"],
                "crop": flag["crop"],
                "crop_sha256": actual_hash,
                "question": flag["question"],
                "prompt": prompt,
                "raw_answer": raw,
                "parsed_answer": parsed,
                "parse_error": parse_error,
                "syntax_normalizations": normalizations,
                "generation_seconds": round(elapsed, 3),
            }
        )
        print(f"[{model_key}] {index}/{len(manifest['flags'])} {flag['label']}: {raw}")

    payload = {
        "plan": PLAN,
        "model_key": model_key,
        "model_id": spec["model_id"],
        "revision": spec["revision"],
        "prompt_version": PROMPT_VERSION,
        "parser_version": PARSER_VERSION,
        "architecture": spec["architecture"],
        "device": "cpu",
        "torch_version": torch.__version__,
        "transformers_generation": {
            "do_sample": False,
            "max_new_tokens": MAX_NEW_TOKENS,
            "seed": SEED,
        },
        "confidence_threshold": manifest["confidence_threshold"],
        "whole_plan_seen": False,
        "coordinates_editable_by_vlm": False,
        "question_order": [item["flag_id"] for item in manifest["flags"]],
        "total_runtime_seconds": round(time.perf_counter() - total_started, 3),
        "answers": answers,
    }
    destination = RESULTS_DIR / spec["result"]
    write_json(destination, payload)
    del model, processor
    gc.collect()
    return destination


def archive_rejected_prompt_outputs() -> None:
    rejected = RESULTS_DIR / "rejected"
    rejected.mkdir(parents=True, exist_ok=True)
    model_paths = [RESULTS_DIR / spec["result"] for spec in MODELS.values()]
    stale_prompt = any(
        path.exists() and read_json(path).get("prompt_version") != PROMPT_VERSION
        for path in model_paths
    )
    if not stale_prompt:
        return
    candidates = [
        *model_paths,
        comparison_path(),
        promoted_path(),
    ]
    for source in candidates:
        if not source.exists():
            continue
        destination = rejected / source.name.replace("-HW3.json", "-prompt-v1-HW3.json")
        shutil.copy2(source, destination)
        source.unlink()


def refresh_parser_outputs() -> None:
    rejected = RESULTS_DIR / "rejected"
    rejected.mkdir(parents=True, exist_ok=True)
    changed = False
    for spec in MODELS.values():
        source = RESULTS_DIR / spec["result"]
        payload = read_json(source)
        if payload.get("parser_version") == PARSER_VERSION:
            continue
        destination = rejected / source.name.replace(
            "-HW3.json", "-prompt-v2-strict-parser-HW3.json"
        )
        shutil.copy2(source, destination)
        for answer in payload["answers"]:
            parsed, parse_error, normalizations = parse_strict_json(answer["raw_answer"])
            answer["parsed_answer"] = parsed
            answer["parse_error"] = parse_error
            answer["syntax_normalizations"] = normalizations
        payload["parser_version"] = PARSER_VERSION
        write_json(source, payload)
        changed = True
    if changed:
        for source in [
            comparison_path(),
            promoted_path(),
        ]:
            if not source.exists():
                continue
            destination = rejected / source.name.replace(
                "-HW3.json", "-prompt-v2-strict-parser-HW3.json"
            )
            shutil.copy2(source, destination)
            source.unlink()


def internally_contradictory(parsed: dict[str, Any]) -> bool:
    reason = parsed["reason"].lower()
    if parsed["continuous"] is True:
        return re.search(r"\bnot(?:\s+\w+){0,3}\s+continuous\b", reason) is not None
    return False


def score_model(
    result: dict[str, Any], manifest: dict[str, Any], reference: dict[str, Any]
) -> dict[str, Any]:
    threshold = float(manifest["confidence_threshold"])
    by_flag = {item["flag_id"]: item for item in manifest["flags"]}
    details: list[dict[str, Any]] = []
    confidence_values: list[float] = []
    error_values: list[float] = []
    resolved = broken = 0
    for item in result["answers"]:
        flag = by_flag[item["flag_id"]]
        expected = reference["walls"][flag["label"]]
        parsed = item["parsed_answer"]
        if parsed is None:
            status = "unresolved"
            reason = item["parse_error"]
            confidence = None
            error = None
        else:
            confidence = float(parsed["confidence"])
            confidence_values.append(confidence)
            error = size_error_mm(parsed["size_mm"], expected["size_mm"])
            if error is not None:
                error_values.append(error)
            eligible = parsed["label"] == flag["label"] and confidence >= threshold
            contradiction = internally_contradictory(parsed)
            agrees = (
                error == 0.0
                and parsed["continuous"] is expected["continuous"]
            )
            if eligible and contradiction:
                status = "unresolved"
                reason = "answer field and visual reason contradict each other"
            elif eligible and agrees:
                status = "resolved"
                reason = "above threshold and agrees with frozen independent reference"
                resolved += 1
            elif eligible:
                status = "broken"
                reason = "above threshold but disagrees with frozen independent reference"
                broken += 1
            else:
                status = "unresolved"
                reason = "below threshold or wrong label"
        details.append(
            {
                "flag_id": item["flag_id"],
                "label": flag["label"],
                "status": status,
                "confidence": confidence,
                "size_error_mm": error,
                "internal_contradiction": (
                    internally_contradictory(parsed) if parsed is not None else False
                ),
                "gate_reason": reason,
            }
        )

    return {
        "model_key": result["model_key"],
        "model_id": result["model_id"],
        "flags_total": len(details),
        "flags_resolved": resolved,
        "flags_broken": broken,
        "flags_unresolved": len(details) - resolved - broken,
        "mean_reported_confidence": (
            round(sum(confidence_values) / len(confidence_values), 6)
            if confidence_values
            else None
        ),
        "mean_size_error_mm": (
            round(sum(error_values) / len(error_values), 3)
            if error_values
            else None
        ),
        "details": details,
    }


def promotion_rank(score: dict[str, Any]) -> tuple[Any, ...]:
    size_error = score["mean_size_error_mm"]
    confidence = score["mean_reported_confidence"]
    return (
        score["flags_broken"] == 0,
        score["flags_resolved"],
        -score["flags_broken"],
        -(size_error if size_error is not None else math.inf),
        confidence if confidence is not None else -math.inf,
        score["model_key"],
    )


def build_promoted_record(
    winner_result: dict[str, Any], winner_score: dict[str, Any]
) -> dict[str, Any]:
    record = copy.deepcopy(read_json(CLASSICAL_PATH))
    record["source"]["promoted_vlm"] = winner_result["model_id"]
    record["source"]["promoted_vlm_revision"] = winner_result["revision"]
    record["method"]["vlm_used_for_coordinates"] = False
    record["method"]["vlm_scope"] = "frozen flag crops only"
    answers = {item["flag_id"]: item for item in winner_result["answers"]}
    scores = {item["flag_id"]: item for item in winner_score["details"]}
    for element in record["elements"]:
        if element["class"] != "wall":
            continue
        label = element["labels"][0]
        flag_id = f"{PLAN}:{label}:continuity"
        if flag_id not in scores or scores[flag_id]["status"] != "resolved":
            continue
        answer = answers[flag_id]["parsed_answer"]
        audit = element["audit"]
        audit["classical_flags"] = list(audit["flags"])
        audit["flags"] = []
        audit["vlm_resolution"] = {
            "source": f'corrected by {winner_result["model_id"]}',
            "revision": winner_result["revision"],
            "confidence": answer["confidence"],
            "answer_size_mm": answer["size_mm"],
            "continuous": answer["continuous"],
            "coordinates_changed": False,
        }
        element["continuous"] = answer["continuous"]
    return record


def score_and_promote(manifest: dict[str, Any]) -> None:
    refresh_parser_outputs()
    reference = read_json(REFERENCE_PATH)
    results = {
        key: read_json(RESULTS_DIR / spec["result"]) for key, spec in MODELS.items()
    }
    scores = [score_model(results[key], manifest, reference) for key in MODELS]
    ranked = sorted(scores, key=promotion_rank, reverse=True)
    winner_score = ranked[0]
    if winner_score["flags_broken"] != 0:
        raise RuntimeError("no model passed the no-regression gate")
    winner_key = winner_score["model_key"]
    comparison = {
        "plan": PLAN,
        "independent_reference": str(REFERENCE_PATH.relative_to(ROOT)).replace("\\", "/"),
        "confidence_threshold": manifest["confidence_threshold"],
        "same_crops_questions_order": True,
        "selection_rule": (
            "zero broken required; then more resolved; ties use lower mean size error, "
            "higher mean reported confidence, then deterministic model key"
        ),
        "models": scores,
        "promoted_model_key": winner_key,
        "promoted_model_id": results[winner_key]["model_id"],
        "gate_passed": True,
    }
    write_json(comparison_path(), comparison)
    promoted = build_promoted_record(results[winner_key], winner_score)
    write_json(promoted_path(), promoted)
    print(
        f"Promoted {winner_key}: {winner_score['flags_resolved']} resolved, "
        f"{winner_score['flags_broken']} broken"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["all", *MODELS],
        default="all",
        help="model run(s) to execute before scoring",
    )
    parser.add_argument(
        "--score-only", action="store_true", help="score existing frozen model outputs"
    )
    args = parser.parse_args()
    manifest = read_json(MANIFEST_PATH)
    if not args.score_only:
        archive_rejected_prompt_outputs()
        keys = list(MODELS) if args.model == "all" else [args.model]
        for key in keys:
            run_model(key, manifest)
    result_paths = [RESULTS_DIR / spec["result"] for spec in MODELS.values()]
    if all(path.exists() for path in result_paths):
        score_and_promote(manifest)
    else:
        missing = ", ".join(path.name for path in result_paths if not path.exists())
        print(f"Scoring deferred until the remaining frozen output exists: {missing}")


if __name__ == "__main__":
    main()
