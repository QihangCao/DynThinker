# Copyright (c) 2026 The DynThinker Authors. Licensed under the MIT License, except
# parse_sampling_rate: adapted from Qwen-VL fine-tuning, Apache License 2.0.
# See THIRD_PARTY_NOTICES.md and LICENSES/Apache-2.0.txt.
"""Dataset registry loaded from a YAML or JSON file.

Each entry maps a dataset name to its annotation file, media root and modality tag::

    spar_234k:
      annotation_path: ${DATA_ROOT}/train/spar_234k.json   # .json (list) or .jsonl
      data_path: ${DATA_ROOT}/media                        # root for relative image/video paths
      tag: 3d                                              # 3d or 2d (used for modality-grouped batching)

Relative paths are resolved against the registry file's directory; ``$VARS`` are expanded.
``--dataset_use`` takes a comma-separated list of names; ``name%N`` keeps a random N% of the samples.
"""

import json
import os
import re
from typing import Dict, List

_VALID_TAGS = ("2d", "3d")


def parse_sampling_rate(dataset_name: str) -> float:
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def _resolve(path: str, base_dir: str) -> str:
    path = os.path.expanduser(os.path.expandvars(str(path)))
    if "$" in path:
        raise ValueError(f"Unresolved environment variable in dataset path: {path!r}")
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(base_dir, path))


def load_registry(config_path: str) -> Dict[str, dict]:
    with open(config_path, "r", encoding="utf-8") as handle:
        if config_path.endswith((".yaml", ".yml")):
            import yaml

            raw = yaml.safe_load(handle)
        else:
            raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Dataset registry {config_path} must be a mapping of name -> entry")
    base_dir = os.path.dirname(os.path.abspath(config_path))
    registry = {}
    for name, entry in raw.items():
        if not isinstance(entry, dict) or "annotation_path" not in entry:
            raise ValueError(f"Dataset {name!r} needs at least 'annotation_path'")
        tag = str(entry.get("tag", "2d"))
        if tag not in _VALID_TAGS:
            raise ValueError(f"Dataset {name!r} has tag {tag!r}; expected one of {_VALID_TAGS}")
        registry[name] = {
            "annotation_path": _resolve(entry["annotation_path"], base_dir),
            "data_path": _resolve(entry.get("data_path", "."), base_dir),
            "tag": tag,
        }
    return registry


def data_list(dataset_names: List[str], registry: Dict[str, dict]) -> List[dict]:
    """Resolve ``name[%N]`` entries against the registry."""
    configs = []
    for dataset_name in dataset_names:
        dataset_name = dataset_name.strip()
        if not dataset_name:
            continue
        sampling_rate = parse_sampling_rate(dataset_name)
        name = re.sub(r"%(\d+)$", "", dataset_name)
        if name not in registry:
            raise ValueError(f"Dataset {name!r} is not defined in the dataset registry")
        config = dict(registry[name])
        config["sampling_rate"] = sampling_rate
        config["dataset_name"] = name
        configs.append(config)
    return configs
