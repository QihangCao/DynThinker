# Prompting and scoring follow the lmms-eval task definitions used by the research evaluation scripts:
# VSI-Bench from thinking-in-space (https://github.com/vision-x-nyu/thinking-in-space, Apache License 2.0)
# and a VLM4D task written for GeoThinker/DynThinker. Reimplemented here without the lmms-eval framework.
"""Prompt construction and scoring for VSI-Bench and VLM4D."""

import ast
import math
import os
import re
from typing import Dict, List

import numpy as np

# ----------------------------------------------------------------------------------------- VSI-Bench
VSI_MCA_QUESTION_TYPES = [
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_distance",
    "route_planning",
    "obj_appearance_order",
]
VSI_NA_QUESTION_TYPES = [
    "object_abs_distance",
    "object_counting",
    "object_size_estimation",
    "room_size_estimation",
]
PRE_PROMPT = "These are frames of a video."
MCA_POST_PROMPT = "Answer with the option's letter from the given choices directly."
NA_POST_PROMPT = "Please answer the question using a single word or phrase."


def vsibench_video_path(doc: dict, media_root: str) -> str:
    return os.path.join(media_root, doc["dataset"], doc["scene_name"] + ".mp4")


def vsibench_prompt(doc: dict) -> str:
    question = doc["question"]
    if doc["question_type"] in VSI_NA_QUESTION_TYPES:
        return PRE_PROMPT + "\n" + question + "\n" + NA_POST_PROMPT
    if doc["question_type"] in VSI_MCA_QUESTION_TYPES:
        options = "Options:\n" + "\n".join(doc["options"])
        return "\n".join([PRE_PROMPT, question, options, MCA_POST_PROMPT])
    raise ValueError(f"Unknown question type: {doc['question_type']}")


def fuzzy_matching(pred: str) -> str:
    return pred.split(" ")[0].rstrip(".").strip()


def exact_match(pred: str, target: str) -> float:
    return 1.0 if pred.lower() == str(target).lower() else 0.0


def mean_relative_accuracy(pred: float, target: float, start: float = 0.5, end: float = 0.95, interval: float = 0.05) -> float:
    """MRA: fraction of thresholds c in {0.5, 0.55, ..., 0.95} with |pred - target| / target <= 1 - c."""
    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    accuracy = abs(pred - target) / target <= 1 - conf_intervs
    return float(accuracy.mean())


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def vsibench_score(doc: dict, prediction: str) -> Dict[str, float]:
    if doc["question_type"] in VSI_MCA_QUESTION_TYPES:
        return {"accuracy": exact_match(fuzzy_matching(prediction), doc["ground_truth"])}
    if doc["question_type"] in VSI_NA_QUESTION_TYPES:
        pred = _to_float(fuzzy_matching(prediction))
        target = _to_float(doc["ground_truth"])
        if pred is None or target is None:
            return {"MRA:.5:.95:.05": 0.0}
        return {"MRA:.5:.95:.05": mean_relative_accuracy(pred, target)}
    raise ValueError(f"Unknown question type: {doc['question_type']}")


def vsibench_aggregate(records: List[dict]) -> Dict[str, float]:
    """Per-type means, relative direction averaged over easy/medium/hard, overall = mean over types (x100)."""
    by_type: Dict[str, List[float]] = {}
    for record in records:
        metric = "accuracy" if record["question_type"] in VSI_MCA_QUESTION_TYPES else "MRA:.5:.95:.05"
        by_type.setdefault(f"{record['question_type']}_{metric}", []).append(record["score"][metric])
    output = {key: float(np.mean(values)) for key, values in by_type.items()}
    direction_keys = [f"object_rel_direction_{level}_accuracy" for level in ("easy", "medium", "hard")]
    if all(key in output for key in direction_keys):
        output["object_rel_direction_accuracy"] = sum(output.pop(key) for key in direction_keys) / 3.0
    output["overall"] = sum(output.values()) / len(output) if output else math.nan
    return {key: value * 100.0 for key, value in output.items()}


# --------------------------------------------------------------------------------------------- VLM4D
def vlm4d_choices(doc: dict) -> Dict[str, str]:
    choices = doc["choices"]
    if isinstance(choices, dict):
        return choices
    if isinstance(choices, str):
        parsed = ast.literal_eval(choices)
        if isinstance(parsed, dict):
            return parsed
    if isinstance(choices, (list, tuple)) and len(choices) == 1 and isinstance(choices[0], str):
        parsed = ast.literal_eval(choices[0])
        if isinstance(parsed, dict):
            return parsed
    raise ValueError(f"Unsupported VLM4D choices payload: {choices!r}")


def vlm4d_video_path(doc: dict, media_root: str) -> str:
    source_path = doc["video"]
    if "resolve/main/" in source_path:
        source_path = source_path.split("resolve/main/", 1)[1]
    return os.path.join(media_root, source_path.lstrip("/"))


def vlm4d_prompt(doc: dict) -> str:
    choices = vlm4d_choices(doc)
    options = "\nOptions:\n" + "\n".join(f"{key}. {value}" for key, value in choices.items())
    return PRE_PROMPT + "\n" + doc["question"] + options + "\n" + MCA_POST_PROMPT


def vlm4d_fuzzy_matching(pred: str) -> str:
    answer_matches = re.findall(r"<answer>(.*?)</answer>", pred, re.DOTALL)
    if answer_matches:
        pred = answer_matches[-1]
    return pred.split(" ")[0].rstrip(".").strip()


def vlm4d_score(doc: dict, prediction: str) -> Dict[str, float]:
    """Correct iff the predicted option letter maps to the ground-truth answer text."""
    choices = vlm4d_choices(doc)
    letter = vlm4d_fuzzy_matching(prediction)
    result = float(doc["answer"] == choices[letter]) if letter in choices else 0.0
    first = letter[:1]
    first_match = float(doc["answer"] == choices[first]) if first in choices else 0.0
    return {"accuracy": result, "first_match_accuracy": first_match}


def vlm4d_aggregate(records: List[dict]) -> Dict[str, float]:
    output = {
        "overall": 100.0 * float(np.mean([r["score"]["accuracy"] for r in records])),
        "first_match_overall": 100.0 * float(np.mean([r["score"]["first_match_accuracy"] for r in records])),
    }
    for type_key in ("type", "question_type"):
        if records and type_key in records[0]:
            groups: Dict[str, List[float]] = {}
            for record in records:
                groups.setdefault(str(record[type_key]), []).append(record["score"]["accuracy"])
            for name, values in groups.items():
                output[f"{type_key}:{name}"] = 100.0 * float(np.mean(values))
            break
    return output


BENCHMARKS = {
    "vsibench": {
        "video_path": vsibench_video_path,
        "prompt": vsibench_prompt,
        "score": vsibench_score,
        "aggregate": vsibench_aggregate,
        "max_new_tokens": 128,
        "keep_fields": ("id", "question_type", "ground_truth"),
    },
    "vlm4d": {
        "video_path": vlm4d_video_path,
        "prompt": vlm4d_prompt,
        "score": vlm4d_score,
        "aggregate": vlm4d_aggregate,
        "max_new_tokens": 16,
        "keep_fields": ("id", "type", "question_type", "answer"),
    },
}
