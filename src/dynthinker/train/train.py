# Adapted from the Qwen-VL fine-tuning code (https://github.com/QwenLM/Qwen3-VL,
# qwen-vl-finetune/qwenvl/train/train_qwen.py, Apache License 2.0), which itself adopts code from
# FastChat and Stanford Alpaca (Apache License 2.0). Restructured and extended by the DynThinker authors
# (model construction, trainable-parameter selection, D-EMA, checkpoint/resume handling).
"""Supervised fine-tuning with DynThinker (VDPM + SDF) and D-EMA.

Launch with torchrun, e.g. ``torchrun --nproc-per-node 8 -m dynthinker.train.train --help``.
"""

import json
import logging
import os
import pathlib
import sys

import torch
import transformers
from transformers import AutoConfig, AutoProcessor, AutoTokenizer, set_seed

from ..data.dataset import make_supervised_data_module
from .arguments import DataArguments, ModelArguments, TrainingArguments
from .dema import DecoupledEMACallback
from .export_ema import EXPORT_MARKER_FILENAME
from .trainer import DynThinkerTrainer

logger = logging.getLogger("dynthinker.train")

# Model arguments recorded in config.json. `geometry_encoder_path` is deliberately not recorded so
# that local paths do not leak into shared checkpoints.
CONFIG_KEYS = (
    "use_geometry_encoder",
    "geometry_encoder_type",
    "feature_fusion_method",
    "cam_merger_type",
    "geometry_merger_type",
    "vdpm_use_decoder_features",
    "vdpm_temporal_mode",
    "vdpm_decoder_layer_offset",
    "vdpm_compute_prediction",
    "vdpm_concat_encoder_decoder_features",
    "exclude_geometry_encoder_on_save",
    "geo_cross_attn",
    "geo_inject_version",
    "geo_importance_gate",
    "depart_smi_token",
    "smi_image_num",
    "smi_downsample_rate",
    "ema_enable",
    "ema_decay",
    "ema_start_step",
    "ema_update_interval",
    "ema_trainable_only",
    "ema_mlp_layer_threshold",
    "ema_mlp_decay",
    "ema_other_decay",
    "ema_new_module_decay",
    "ema_mlp_save_as_final",
    "ema_other_save_as_final",
    "ema_new_module_save_as_final",
)

GEOMETRY_ENCODER_STATE_PREFIXES = (
    "model.geometry_encoder.",
    "geometry_encoder.",
    "module.model.geometry_encoder.",
    "module.geometry_encoder.",
)


def detect_backbone(model_name_or_path: str) -> str:
    """Return "qwen3vl" or "qwen2.5vl" from the checkpoint's config."""
    config = AutoConfig.from_pretrained(model_name_or_path)
    model_type = getattr(config, "model_type", "")
    if model_type == "qwen3_vl":
        return "qwen3vl"
    if model_type == "qwen2_5_vl":
        return "qwen2.5vl"
    raise ValueError(f"Unsupported backbone model_type={model_type!r}; expected qwen3_vl or qwen2_5_vl")


def apply_model_args_to_config(config, model_args, include_text_config: bool) -> None:
    for key in CONFIG_KEYS:
        value = getattr(model_args, key)
        setattr(config, key, value)
        if include_text_config and hasattr(config, "text_config"):
            setattr(config.text_config, key, value)


def strip_geometry_encoder_from_state_dict(state_dict):
    filtered = {k: v for k, v in state_dict.items() if not k.startswith(GEOMETRY_ENCODER_STATE_PREFIXES)}
    return filtered, len(state_dict) - len(filtered)


class GeoAwareTrainer(DynThinkerTrainer):
    """Optionally omits the (frozen) VDPM weights from saved checkpoints."""

    def _save(self, output_dir=None, state_dict=None):
        if not getattr(self.model, "_exclude_geometry_encoder_on_save", False):
            return super()._save(output_dir=output_dir, state_dict=state_dict)
        if state_dict is None:
            state_dict = self.model.state_dict()
        state_dict, removed = strip_geometry_encoder_from_state_dict(state_dict)
        if removed and self.args.should_save:
            logger.info("Skip saving geometry encoder params: %d keys", removed)
        return super()._save(output_dir=output_dir, state_dict=state_dict)


def set_trainable(model_args, model, backbone: str) -> None:
    """Select trainable parameters. SDF (`cross_attn`) lives inside the language model."""
    inner = model.model
    visual = inner.visual
    language_model = inner.language_model

    for p in visual.parameters():
        p.requires_grad = bool(model_args.tune_mm_vision)
    if hasattr(visual, "merger"):
        for p in visual.merger.parameters():
            p.requires_grad = bool(model_args.tune_mm_mlp)

    if model_args.tune_mm_llm:
        for p in language_model.parameters():
            p.requires_grad = True
        for p in model.lm_head.parameters():
            p.requires_grad = True
    else:
        for n, p in language_model.named_parameters():
            p.requires_grad = "cross_attn" in n
        for p in model.lm_head.parameters():
            p.requires_grad = False

    if model_args.use_geometry_encoder:
        for p in inner.geometry_encoder.parameters():
            p.requires_grad = False
    if hasattr(inner, "geometry_merger"):
        tune = bool(model_args.tune_geometry_merger) or model_args.feature_fusion_method != "zero"
        for p in inner.geometry_merger.parameters():
            p.requires_grad = tune


def log_trainable_summary(model) -> None:
    groups = {"visual": 0, "visual_merger": 0, "llm": 0, "sdf": 0, "lm_head": 0, "geometry_merger": 0, "other": 0}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        n = param.numel()
        if "visual.merger" in name:
            groups["visual_merger"] += n
        elif ".visual." in name or name.startswith("visual."):
            groups["visual"] += n
        elif "geometry_merger" in name:
            groups["geometry_merger"] += n
        elif "cross_attn" in name:
            groups["sdf"] += n
        elif "lm_head" in name:
            groups["lm_head"] += n
        elif "language_model" in name:
            groups["llm"] += n
        else:
            groups["other"] += n
    total = sum(groups.values())
    logger.info("Trainable parameters: %d (%.3f M)", total, total / 1e6)
    for key, value in groups.items():
        logger.info("  %-16s %d (%.3f M)", key, value, value / 1e6)


def build_model(model_args, training_args, data_args, attn_implementation):
    backbone = detect_backbone(model_args.model_name_or_path)
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    common = dict(
        config=config,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_implementation,
        torch_dtype=torch.bfloat16,
    )

    if backbone == "qwen2.5vl":
        if model_args.use_geometry_encoder or model_args.geo_cross_attn:
            raise ValueError(
                "VDPM + SDF is only available for Qwen3-VL in this release; "
                "Qwen2.5-VL supports D-EMA training without SDF (--use_geometry_encoder false)."
            )
        apply_model_args_to_config(config, model_args, include_text_config=False)
        from transformers import Qwen2_5_VLForConditionalGeneration

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_args.model_name_or_path, **common)
    else:
        apply_model_args_to_config(config, model_args, include_text_config=True)
        if model_args.use_geometry_encoder:
            if not model_args.geometry_encoder_path:
                raise ValueError("--geometry_encoder_path is required with --use_geometry_encoder true")
            from ..model.modeling_qwen3_vl import Qwen3VLForConditionalGenerationWithVGGT

            model = Qwen3VLForConditionalGenerationWithVGGT.from_pretrained(model_args.model_name_or_path, **common)
            model.load_vdpm_weights(model_args.geometry_encoder_path)
            if getattr(config, "depart_smi_token", False):
                data_args.depart_smi_token = True
                data_args.smi_image_num = getattr(config, "smi_image_num", 8)
                data_args.smi_downsample_rate = getattr(config, "smi_downsample_rate", 2)
        else:
            if model_args.geo_cross_attn:
                raise ValueError("--geo_cross_attn requires --use_geometry_encoder true")
            from transformers import Qwen3VLForConditionalGeneration

            model = Qwen3VLForConditionalGeneration.from_pretrained(model_args.model_name_or_path, **common)

    # The dataset adjusts the pixel limits of `processor.image_processor` in place; the trainer saves the
    # adjusted image processor, the video processor and the chat template with every checkpoint.
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)
    data_args.image_processor = processor.image_processor
    data_args.model_type = backbone
    return model, backbone, processor


def _reject_exported_model(path: pathlib.Path) -> None:
    if (path / EXPORT_MARKER_FILENAME).is_file():
        raise ValueError(
            f"{path} is a final or exported inference model (possibly D-EMA averaged weights, no optimizer or "
            "EMA state) and cannot be resumed; resume from a checkpoint-<step> directory instead"
        )


def find_resume_checkpoint(training_args, ema_callback):
    """Explicit --resume_from_checkpoint wins; otherwise (auto_resume) the latest checkpoint-<step> in output_dir."""
    if training_args.resume_from_checkpoint:
        path = pathlib.Path(training_args.resume_from_checkpoint)
        if not path.is_dir():
            raise FileNotFoundError(f"--resume_from_checkpoint {path} does not exist")
        _reject_exported_model(path)
        return str(path)
    if not training_args.auto_resume:
        return None

    candidates = [
        p
        for p in pathlib.Path(training_args.output_dir).glob("checkpoint-*")
        if p.is_dir() and p.name.split("-", 1)[1].isdigit() and not (p / EXPORT_MARKER_FILENAME).is_file()
    ]
    if ema_callback is not None and ema_callback.checkpoint_save and ema_callback.checkpoint_strict:
        complete = [p for p in candidates if ema_callback.has_checkpoint_state(p)]
        skipped = sorted(p.name for p in candidates if p not in complete)
        if skipped:
            logger.warning("[D-EMA] skipping checkpoints without complete EMA state: %s", ", ".join(skipped))
        if candidates and not complete:
            raise RuntimeError(
                "EMA strict resume found checkpoint directories but none has a complete EMA shadow; "
                "refusing to restart from the base model"
            )
        candidates = complete

    candidates = sorted(candidates, key=lambda path: int(path.name.split("-", 1)[1]))
    return str(candidates[-1]) if candidates else None


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        level=logging.INFO if training_args.should_log else logging.WARNING,
    )

    set_seed(training_args.seed)
    os.makedirs(training_args.output_dir, exist_ok=True)

    model, backbone, processor = build_model(model_args, training_args, data_args, model_args.attn_implementation)
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer_kwargs = dict(
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=False, **tokenizer_kwargs)
    except Exception as error:  # some checkpoints only ship a fast tokenizer
        logger.warning("Slow tokenizer unavailable (%s); using the default tokenizer class", error)
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)

    set_trainable(model_args, model, backbone)
    if model_args.exclude_geometry_encoder_on_save:
        model._exclude_geometry_encoder_on_save = True
    log_trainable_summary(model)

    data_args.use_geometry_encoder = model_args.use_geometry_encoder
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    callbacks = []
    ema_callback = None
    if model_args.ema_enable:
        ema_callback = DecoupledEMACallback(
            mlp_decay=model_args.ema_mlp_decay,
            other_decay=model_args.ema_other_decay,
            new_module_decay=model_args.ema_new_module_decay,
            mlp_layer_threshold=model_args.ema_mlp_layer_threshold,
            start_step=model_args.ema_start_step,
            update_interval=model_args.ema_update_interval,
            trainable_only=model_args.ema_trainable_only,
            mlp_save_as_final=model_args.ema_mlp_save_as_final,
            other_save_as_final=model_args.ema_other_save_as_final,
            new_module_save_as_final=model_args.ema_new_module_save_as_final,
            decay=model_args.ema_decay,
            on_gpu=model_args.ema_on_gpu,
            rank_zero_only=model_args.ema_rank_zero_only,
            checkpoint_save=model_args.ema_checkpoint_save,
            checkpoint_strict=model_args.ema_checkpoint_strict,
            checkpoint_dtype=model_args.ema_checkpoint_dtype,
            reinit_on_resume=model_args.ema_reinit_on_resume,
        )
        callbacks.append(ema_callback)

    trainer_cls = GeoAwareTrainer if model_args.exclude_geometry_encoder_on_save else DynThinkerTrainer
    trainer = trainer_cls(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        callbacks=callbacks,
        processor=processor,
        **data_module,
    )

    resume_checkpoint = find_resume_checkpoint(training_args, ema_callback)
    if resume_checkpoint is not None:
        logger.info("Resuming from checkpoint: %s", resume_checkpoint)
        if ema_callback is not None:
            ema_callback.set_resume_checkpoint(resume_checkpoint)
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    else:
        trainer.train()

    trainer.save_state()

    model.config.use_cache = True
    if ema_callback is not None:
        applied = ema_callback.maybe_apply_to_model(trainer.model)
        if applied > 0:
            logger.info("[D-EMA] applied EMA weights before final save, tensors=%d", applied)

    save_final_model(trainer, training_args.output_dir, model_args.exclude_geometry_encoder_on_save)
    if training_args.should_save:
        final_groups = ema_callback.final_groups() if ema_callback is not None else []
        write_final_marker(training_args.output_dir, final_groups)


def write_final_marker(output_dir: str, ema_groups) -> None:
    """Mark the final model directory as an inference model that `dynthinker-train` must not resume."""
    with open(os.path.join(output_dir, EXPORT_MARKER_FILENAME), "w", encoding="utf-8") as handle:
        json.dump({"source_checkpoint": "final", "groups": list(ema_groups), "resumable": False}, handle, indent=2)


def save_final_model(trainer, output_dir: str, exclude_geometry_encoder_on_save: bool) -> None:
    """Save the (D-EMA consolidated) model and the processor files. With DeepSpeed the rank-0 weights are written."""
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return
    state_dict = trainer.model.state_dict()
    if exclude_geometry_encoder_on_save:
        state_dict, removed = strip_geometry_encoder_from_state_dict(state_dict)
        if trainer.args.should_save and removed:
            logger.info("Skip saving geometry encoder params: %d keys", removed)
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


def main():
    train()


if __name__ == "__main__":
    sys.exit(main())
