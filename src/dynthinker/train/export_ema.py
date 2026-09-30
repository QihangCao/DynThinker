# Copyright (c) 2026 The DynThinker Authors. Licensed under the MIT License.
"""Export D-EMA weights from an intermediate Trainer checkpoint.

Trainer checkpoints (``checkpoint-<step>/``) hold the *online* weights plus the D-EMA shadows
(``ema_state.safetensors`` / ``ema_state.json``). The final model written at the end of training
already contains the consolidated weights; use this tool to obtain the same consolidation for an
intermediate checkpoint. It replaces, tensor by tensor, the weights of the groups that are applied at
the end of training (by default the groups whose ``*_save_as_final`` flag is set in the saved D-EMA
config: ``mlp`` and ``other``; ``new_module`` keeps its online values) and copies the config,
tokenizer and processor files.

``dynthinker-train`` saves the processor files (``preprocessor_config.json``,
``video_preprocessor_config.json`` and the processor chat template ``chat_template.json``) in every
checkpoint. Checkpoints written by earlier versions lack them; the command-line tool then stops before
writing anything. For such a checkpoint, pass ``--processor-from`` with the directory whose processor
was used for training: the final output directory of the same run (it holds the training pixel limits;
if it has no ``video_preprocessor_config.json``, transformers derives the video processor from the image
processor config) or the base model directory.

The export is an inference model, not a training checkpoint: optimizer/scheduler/RNG/trainer state and
the EMA shadows are not copied, and a ``dema_export.json`` marker is written. ``dynthinker-train``
refuses to resume from a directory carrying this marker, because continuing training from averaged
weights with a stale optimizer and without the shadows would silently change the D-EMA trajectory.
Resume from the original ``checkpoint-<step>`` instead.

Usage:
    dynthinker-export-ema --checkpoint OUTPUT/checkpoint-2000 --output EXPORTS/dema-step2000
    dynthinker-export-ema --checkpoint OLD/checkpoint-2000 --output EXPORTS/dema-step2000 --processor-from OLD
"""

import argparse
import json
import os
import shutil
import sys
from typing import Dict, Iterable, List, Optional

import torch

EMA_STATE_FILENAME = "ema_state.safetensors"
EMA_META_FILENAME = "ema_state.json"
EXPORT_MARKER_FILENAME = "dema_export.json"
# Processor files needed next to the tokenizer files to load the multimodal processor of an export.
PROCESSOR_FILENAMES = ("preprocessor_config.json", "video_preprocessor_config.json", "chat_template.json")
_REQUIRED_PROCESSOR_FILENAME = "preprocessor_config.json"
# Resume-only files that must not be carried over into an exported (averaged) model.
_RESUME_ONLY_PREFIXES = (
    "optimizer", "scheduler", "rng_state", "global_step", "zero_to_fp32", "latest", "trainer_state", "scaler",
)


def _final_groups_from_config(config: dict) -> List[str]:
    groups = []
    if config.get("mlp_save_as_final", True):
        groups.append("mlp")
    if config.get("other_save_as_final", True):
        groups.append("other")
    if config.get("new_module_save_as_final", False):
        groups.append("new_module")
    return groups


def _final_groups_from_metadata(checkpoint_dir: str) -> List[str]:
    with open(os.path.join(checkpoint_dir, EMA_META_FILENAME), "r", encoding="utf-8") as handle:
        return _final_groups_from_config(json.load(handle).get("config", {}))


def _weight_files(checkpoint_dir: str) -> List[str]:
    index_path = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path, "r", encoding="utf-8") as handle:
            index = json.load(handle)
        return sorted(set(index["weight_map"].values()))
    single = os.path.join(checkpoint_dir, "model.safetensors")
    if os.path.isfile(single):
        return ["model.safetensors"]
    raise FileNotFoundError(f"No safetensors model weights found in {checkpoint_dir}")


def load_ema_shadows(checkpoint_dir: str, groups: Optional[Iterable[str]] = None) -> Dict[str, torch.Tensor]:
    """Return ``{parameter_name: shadow_tensor}`` for the requested (default: final) groups."""
    from safetensors import safe_open

    meta_path = os.path.join(checkpoint_dir, EMA_META_FILENAME)
    state_path = os.path.join(checkpoint_dir, EMA_STATE_FILENAME)
    if not (os.path.isfile(meta_path) and os.path.isfile(state_path)):
        raise FileNotFoundError(f"{checkpoint_dir} has no {EMA_STATE_FILENAME}/{EMA_META_FILENAME}")
    with open(meta_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("version") != 1:
        raise ValueError(f"Unsupported EMA state version: {metadata.get('version')}")
    if os.path.getsize(state_path) != int(metadata.get("state_file_size", -1)):
        raise ValueError(f"EMA state file size does not match its metadata: {state_path}")
    groups = list(groups) if groups is not None else _final_groups_from_config(metadata.get("config", {}))
    shadows = {}
    with safe_open(state_path, framework="pt", device="cpu") as reader:
        for group in groups:
            for name in metadata["groups"].get(group, []):
                if name in shadows:
                    raise ValueError(f"Parameter {name} appears in more than one EMA group")
                shadows[name] = reader.get_tensor(f"{group}:{name}")
    return shadows


def _check_processor_source(checkpoint_dir: str, processor_from: Optional[str], require_processor: bool) -> None:
    has_processor = os.path.isfile(os.path.join(checkpoint_dir, _REQUIRED_PROCESSOR_FILENAME))
    if processor_from is not None:
        if has_processor:
            raise ValueError(
                f"{checkpoint_dir} already contains its processor files; --processor-from is only for "
                "checkpoints saved without them"
            )
        if not os.path.isfile(os.path.join(processor_from, _REQUIRED_PROCESSOR_FILENAME)):
            raise FileNotFoundError(f"--processor-from {processor_from} has no {_REQUIRED_PROCESSOR_FILENAME}")
    elif require_processor and not has_processor:
        raise FileNotFoundError(
            f"{checkpoint_dir} has no processor files ({_REQUIRED_PROCESSOR_FILENAME}), so an export could not "
            "be loaded for inference. Checkpoints written by earlier versions of dynthinker-train lack them: "
            "pass --processor-from with the final output directory of the same run or, if there is none, the "
            "base model directory used for training"
        )


def export_ema_checkpoint(
    checkpoint_dir: str,
    output_dir: str,
    groups: Optional[Iterable[str]] = None,
    processor_from: Optional[str] = None,
    require_processor: bool = False,
) -> Dict[str, int]:
    """Write ``output_dir`` = ``checkpoint_dir`` with D-EMA shadows applied to the weights.

    ``processor_from`` supplies the processor files (``PROCESSOR_FILENAMES`` present there) for a checkpoint
    saved without them. With ``require_processor`` (always set by the command-line tool) a checkpoint without
    processor files and without ``processor_from`` is rejected before anything is written.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    checkpoint_dir = os.path.abspath(checkpoint_dir)
    output_dir = os.path.abspath(output_dir)
    if checkpoint_dir == output_dir:
        raise ValueError("--output must differ from --checkpoint")
    if os.path.exists(output_dir) and os.listdir(output_dir):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    if processor_from is not None:
        processor_from = os.path.abspath(processor_from)
    _check_processor_source(checkpoint_dir, processor_from, require_processor)

    groups = list(groups) if groups is not None else None
    shadows = load_ema_shadows(checkpoint_dir, groups)
    weight_files = _weight_files(checkpoint_dir)
    os.makedirs(output_dir, exist_ok=True)

    replaced = set()
    for filename in weight_files:
        tensors = {}
        with safe_open(os.path.join(checkpoint_dir, filename), framework="pt", device="cpu") as reader:
            metadata = reader.metadata()
            for key in reader.keys():
                tensor = reader.get_tensor(key)
                if key in shadows:
                    shadow = shadows[key]
                    if tuple(shadow.shape) != tuple(tensor.shape):
                        raise ValueError(f"Shape mismatch for {key}: weights {tuple(tensor.shape)} vs EMA {tuple(shadow.shape)}")
                    tensor = shadow.to(dtype=tensor.dtype)
                    replaced.add(key)
                tensors[key] = tensor.contiguous()
        save_file(tensors, os.path.join(output_dir, filename), metadata=metadata)

    missing = sorted(set(shadows) - replaced)
    if missing:
        shutil.rmtree(output_dir)
        raise KeyError(
            f"{len(missing)} EMA tensors have no matching weight in the checkpoint (e.g. {missing[:3]}); "
            "nothing was written"
        )

    weight_set = set(weight_files)
    for entry in os.listdir(checkpoint_dir):
        source = os.path.join(checkpoint_dir, entry)
        if entry in weight_set or entry in (EMA_STATE_FILENAME, EMA_META_FILENAME, EXPORT_MARKER_FILENAME):
            continue
        if entry.startswith(_RESUME_ONLY_PREFIXES) or os.path.isdir(source):
            continue
        shutil.copy2(source, os.path.join(output_dir, entry))
    if processor_from is not None:
        for name in PROCESSOR_FILENAMES:
            if os.path.isfile(os.path.join(processor_from, name)):
                shutil.copy2(os.path.join(processor_from, name), os.path.join(output_dir, name))

    with open(os.path.join(output_dir, EXPORT_MARKER_FILENAME), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "source_checkpoint": os.path.basename(checkpoint_dir),
                "replaced_tensors": len(replaced),
                "groups": groups if groups is not None else _final_groups_from_metadata(checkpoint_dir),
                "resumable": False,
            },
            handle,
            indent=2,
        )
    return {"replaced_tensors": len(replaced), "weight_files": len(weight_files)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="dynthinker-export-ema",
        description="Apply saved D-EMA shadows to the weights of an intermediate Trainer checkpoint.",
    )
    parser.add_argument("--checkpoint", required=True, help="Trainer checkpoint directory (checkpoint-<step>)")
    parser.add_argument("--output", required=True, help="New directory for the exported model")
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=["mlp", "other", "new_module"],
        default=None,
        help="EMA groups to apply (default: the *_save_as_final groups recorded in ema_state.json)",
    )
    parser.add_argument(
        "--processor-from",
        default=None,
        help="Only for checkpoints saved without processor files: directory to copy "
        + ", ".join(PROCESSOR_FILENAMES)
        + " from (the final output directory of the same run, or the base model used for training)",
    )
    args = parser.parse_args(argv)
    summary = export_ema_checkpoint(
        args.checkpoint, args.output, args.groups, processor_from=args.processor_from, require_processor=True
    )
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
