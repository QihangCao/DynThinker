# Adapted from the Qwen2.5-VL / Qwen3-VL fine-tuning code (https://github.com/QwenLM/Qwen3-VL,
# qwen-vl-finetune/qwenvl/train/trainer.py, Apache License 2.0) and the length-grouped samplers of LLaVA
# (https://github.com/haotian-liu/LLaVA) and LLaVA-NeXT (https://github.com/LLaVA-VL/LLaVA-NeXT,
# `get_length_grouped_indices_auto_single`), both Apache License 2.0.
# DynThinker changes: the optimizer grouping and sampler are provided by a Trainer subclass instead
# of global monkey patches; the sequence-packing attention patch is not included.
"""Trainer with per-module learning rates and modality-grouped batching."""

import json
import os
from typing import List, Optional

import torch
import torch.nn as nn
from torch.utils.data import Sampler
from transformers import Trainer
from transformers.trainer import get_parameter_names, has_length
from transformers.trainer_pt_utils import get_length_grouped_indices as get_length_grouped_indices_hf

# transformers 4.57 no longer exposes ALL_LAYERNORM_LAYERS from transformers.trainer; the original
# code then fell back to [nn.LayerNorm]. Only nn.LayerNorm weights (e.g. SDF norms) and biases are
# excluded from weight decay; Qwen RMSNorm weights are decayed. Kept as-is for parity.
LAYERNORM_LAYERS = [nn.LayerNorm]


# --------------------------------------------------------------------------- sampler
def split_to_even_chunks(indices, lengths, num_chunks):
    """Split a list of indices into `num_chunks` chunks of roughly equal lengths."""
    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks
    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")
    return chunks


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    """Randomly permute, cut into mega-batches of `world_size * batch_size` and sort each by length."""
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]
    return [i for megabatch in megabatches for batch in megabatch for i in batch]


def get_length_grouped_indices_auto_single(lengths, batch_size, world_size, generator=None):
    indices = get_length_grouped_indices_hf(lengths, batch_size * world_size, generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size] for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]
    batch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in batch_indices]
    return [i for megabatch in megabatches for batch in megabatch for i in batch]


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    """Group samples so that a mega-batch never mixes modalities.

    `lengths` are positive for "3d"-tagged samples and negative for "2d"-tagged samples. As in the
    original implementation, the last partial mega-batch of each modality is dropped per epoch.
    """
    assert all(length != 0 for length in lengths), "Should not have zero length."
    if all(length > 0 for length in lengths) or all(length < 0 for length in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, length) for i, length in enumerate(lengths) if length > 0])
    lang_indices, lang_lengths = zip(*[(i, -length) for i, length in enumerate(lengths) if length < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]
    return [i for megabatch in megabatches for i in megabatch]


class LengthGroupedSampler(Sampler):
    """Sample indices grouped by (modality and) length, with some randomness."""

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")
        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(
                self.lengths, self.batch_size, self.world_size, generator=self.generator
            )
        else:
            indices = get_length_grouped_indices_auto_single(
                self.lengths, self.batch_size, self.world_size, generator=self.generator
            )
        return iter(indices)


# --------------------------------------------------------------------------- processor files
PROCESSOR_CHAT_TEMPLATE_FILENAME = "chat_template.json"


def save_processor_files(processor, output_dir: str) -> None:
    """Write the image/video processor configs and the processor chat template (not the tokenizer).

    The template goes to ``chat_template.json``, which ``ProcessorMixin.from_pretrained`` reads in
    preference to the tokenizer's ``chat_template.jinja``, so a processor loaded from ``output_dir``
    renders prompts exactly like ``processor``.
    """
    processor.image_processor.save_pretrained(output_dir)
    video_processor = getattr(processor, "video_processor", None)
    if video_processor is not None:
        video_processor.save_pretrained(output_dir)
    if processor.chat_template is not None:
        with open(os.path.join(output_dir, PROCESSOR_CHAT_TEMPLATE_FILENAME), "w", encoding="utf-8") as handle:
            handle.write(json.dumps({"chat_template": processor.chat_template}, indent=2, sort_keys=True) + "\n")


# --------------------------------------------------------------------------- trainer
class DynThinkerTrainer(Trainer):
    """Trainer with separate learning rates for the vision tower / mergers and modality grouping.

    If ``processor`` (the multimodal processor of the base model) is given, its image/video processor
    configs and chat template are saved with every model save, i.e. in each ``checkpoint-<step>`` and in
    the final output directory, next to the tokenizer saved as ``processing_class``.
    """

    def __init__(self, *args, processor=None, **kwargs):
        if processor is not None and not isinstance(processor.chat_template, (str, type(None))):
            raise ValueError("Processors with several named chat templates are not supported")
        super().__init__(*args, **kwargs)
        self.processor = processor

    def _save(self, output_dir=None, state_dict=None):
        super()._save(output_dir=output_dir, state_dict=state_dict)
        if self.processor is not None and self.args.should_save:
            save_processor_files(self.processor, output_dir if output_dir is not None else self.args.output_dir)

    def _get_train_sampler(self, train_dataset=None) -> Optional[Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None
        # world_size * gradient_accumulation_steps mirrors the original implementation.
        world_size = self.args.world_size * self.args.gradient_accumulation_steps
        if getattr(self.args, "group_by_length", False):
            return LengthGroupedSampler(self.args.train_batch_size, world_size, lengths=self.train_dataset.lengths)
        if getattr(self.args, "group_by_modality_length", False):
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size,
                lengths=self.train_dataset.modality_lengths,
                group_by_modality=True,
            )
        return super()._get_train_sampler(train_dataset)

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        opt_model = self.model
        decay_parameters = get_parameter_names(opt_model, LAYERNORM_LAYERS)
        decay_parameters = {name for name in decay_parameters if "bias" not in name}
        named = list(opt_model.named_parameters())
        weight_decay = self.args.weight_decay

        def params(select):
            return [p for n, p in named if p.requires_grad and select(n)]

        if self.args.mm_projector_lr is not None and self.args.mm_projector_lr != 0:
            # Note: "merger" matches both the vision merger and the geometry merger.
            projector = {n for n, _ in named if "merger" in n}
            if self.args.vision_tower_lr is not None and self.args.vision_tower_lr != 0:
                vision = {n for n, _ in named if "visual" in n}
                groups = [
                    {"params": params(lambda n: n in decay_parameters and n not in projector and n not in vision),
                     "weight_decay": weight_decay},
                    {"params": params(lambda n: n in decay_parameters and n not in projector and n in vision),
                     "weight_decay": weight_decay, "lr": self.args.vision_tower_lr},
                    {"params": params(lambda n: n not in decay_parameters and n not in projector and n not in vision),
                     "weight_decay": 0.0},
                    {"params": params(lambda n: n not in decay_parameters and n not in projector and n in vision),
                     "weight_decay": 0.0, "lr": self.args.vision_tower_lr},
                    {"params": params(lambda n: n in decay_parameters and n in projector),
                     "weight_decay": weight_decay, "lr": self.args.mm_projector_lr},
                    {"params": params(lambda n: n not in decay_parameters and n in projector),
                     "weight_decay": 0.0, "lr": self.args.mm_projector_lr},
                ]
            else:
                groups = [
                    {"params": params(lambda n: n in decay_parameters and n not in projector),
                     "weight_decay": weight_decay},
                    {"params": params(lambda n: n not in decay_parameters and n not in projector),
                     "weight_decay": 0.0},
                    {"params": params(lambda n: n in decay_parameters and n in projector),
                     "weight_decay": weight_decay, "lr": self.args.mm_projector_lr},
                    {"params": params(lambda n: n not in decay_parameters and n in projector),
                     "weight_decay": 0.0, "lr": self.args.mm_projector_lr},
                ]
        else:
            groups = [
                {"params": params(lambda n: n in decay_parameters), "weight_decay": weight_decay},
                {"params": params(lambda n: n not in decay_parameters), "weight_decay": 0.0},
            ]

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

        # transformers 4.57 parses `--optim_args` but does not forward them to torch AdamW. Allow
        # `--optim_args foreach=false` to select the (mathematically equivalent) single-tensor path.
        parsed_optim_args = {}
        if self.args.optim_args:
            for mapping in self.args.optim_args.replace(" ", "").split(","):
                key, value = mapping.split("=", 1)
                parsed_optim_args[key] = value
        if "foreach" in parsed_optim_args:
            value = parsed_optim_args["foreach"].lower()
            if value not in {"true", "false"}:
                raise ValueError(f"AdamW optim_args foreach must be true or false, got {parsed_optim_args['foreach']!r}")
            optimizer_kwargs["foreach"] = value == "true"

        self.optimizer = optimizer_cls(groups, **optimizer_kwargs)
        return self.optimizer
