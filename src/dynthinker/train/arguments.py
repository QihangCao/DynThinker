# Adapted from the Qwen-VL fine-tuning code (https://github.com/QwenLM/Qwen3-VL,
# qwen-vl-finetune/qwenvl/train/argument.py, Apache License 2.0); `group_by_modality_length` follows
# LLaVA (Apache License 2.0). Extended by the DynThinker authors with the VDPM/SDF and D-EMA options
# of the released configuration.
"""Command-line arguments for `dynthinker-train`."""

from dataclasses import dataclass, field
from typing import Optional

import transformers


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="Qwen/Qwen3-VL-8B-Instruct",
        metadata={"help": "Local directory or Hub id of a Qwen3-VL (or Qwen2.5-VL, D-EMA only) checkpoint"},
    )
    attn_implementation: str = field(
        default="flash_attention_2", metadata={"help": "flash_attention_2 (research code default) or sdpa"}
    )
    tune_mm_llm: bool = field(default=False, metadata={"help": "Train the language model (incl. SDF and lm_head)"})
    tune_mm_mlp: bool = field(default=False, metadata={"help": "Train the vision-language merger"})
    tune_mm_vision: bool = field(default=False, metadata={"help": "Train the vision tower"})
    tune_geometry_merger: bool = field(
        default=False,
        metadata={"help": "Train the geometry feature merger (frozen in the released configuration)"},
    )

    # Geometry encoder (VDPM) and Spatial-Dynamic Fusion
    use_geometry_encoder: bool = field(default=False, metadata={"help": "Enable VDPM + SDF (Qwen3-VL only)"})
    geometry_encoder_type: str = field(default="vdpm")
    geometry_encoder_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the VDPM state dict (model.pt)"}
    )
    feature_fusion_method: str = field(default="zero")
    cam_merger_type: str = field(default="zero")
    geometry_merger_type: str = field(default="mlp")
    vdpm_use_decoder_features: bool = field(default=True)
    vdpm_temporal_mode: str = field(default="diag")
    vdpm_decoder_layer_offset: int = field(default=-2)
    vdpm_compute_prediction: bool = field(default=False)
    vdpm_concat_encoder_decoder_features: bool = field(default=True)
    exclude_geometry_encoder_on_save: bool = field(
        default=False,
        metadata={"help": "Do not store VDPM weights in saved checkpoints (they must then be passed at load time)"},
    )
    geo_cross_attn: bool = field(default=False, metadata={"help": "Insert SDF modules into the middle LLM layers"})
    geo_inject_version: str = field(default="v1", metadata={"help": "SDF variant; only 'v1' is supported"})
    geo_importance_gate: bool = field(
        default=True, metadata={"help": "Recorded in the config for compatibility; SDF always uses importance gating"}
    )

    # Visual token compression: spatial merge size x `smi_downsample_rate` when a sample has more than
    # `smi_image_num` frames.
    depart_smi_token: bool = field(default=False)
    smi_image_num: int = field(default=8)
    smi_downsample_rate: int = field(default=2)

    # D-EMA
    ema_enable: bool = field(default=False, metadata={"help": "Enable D-EMA"})
    ema_decay: float = field(default=0.999, metadata={"help": "Fallback decay for other params when ema_other_decay < 0"})
    ema_start_step: int = field(default=200, metadata={"help": "First optimizer step with an EMA update (0 behaves like ema_update_interval)"})
    ema_update_interval: int = field(default=1, metadata={"help": "EMA update period in optimizer steps"})
    ema_trainable_only: bool = field(default=True)
    ema_mlp_layer_threshold: int = field(
        default=4, metadata={"help": "Late-MLP group: 0-based decoder layer index >= this value (research code rule)"}
    )
    ema_mlp_decay: float = field(default=-1.0, metadata={"help": "Decay of the late-MLP (sensitive) group; required"})
    ema_other_decay: float = field(default=-1.0, metadata={"help": "Decay of other pretrained params; <0 uses ema_decay"})
    ema_new_module_decay: float = field(
        default=-1.0, metadata={"help": "Shadow decay of new modules; <0 uses the other decay"}
    )
    ema_mlp_save_as_final: bool = field(default=True)
    ema_other_save_as_final: bool = field(default=True)
    ema_new_module_save_as_final: bool = field(
        default=False, metadata={"help": "Keep False: new modules (SDF) keep their online weights"}
    )
    ema_on_gpu: bool = field(default=False, metadata={"help": "Keep EMA shadows on GPU"})
    ema_rank_zero_only: bool = field(
        default=False, metadata={"help": "Keep shadows only on global rank 0 (valid for synchronized data parallelism)"}
    )
    ema_checkpoint_save: bool = field(
        default=False,
        metadata={"help": "Save EMA shadows with Trainer checkpoints (needed to resume with D-EMA)"},
    )
    ema_checkpoint_strict: bool = field(
        default=True, metadata={"help": "Fail on resume if the EMA state is missing or inconsistent"}
    )
    ema_checkpoint_dtype: str = field(default="float32", metadata={"help": "float32, bfloat16 or float16"})
    ema_reinit_on_resume: bool = field(
        default=False,
        metadata={"help": "Allow resuming without saved EMA state (shadows restart from the resumed weights)"},
    )


@dataclass
class DataArguments:
    dataset_use: str = field(
        default="", metadata={"help": "Comma-separated dataset names from --dataset_config, optional '%%N' sampling"}
    )
    dataset_config: Optional[str] = field(
        default=None, metadata={"help": "YAML/JSON dataset registry (see configs/datasets.example.yaml)"}
    )
    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    base_interval: int = field(default=2, metadata={"help": "Seconds per sampled frame before clamping"})
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)
    video_max_frame_pixels: int = field(default=32 * 28 * 28)
    video_min_frame_pixels: int = field(default=4 * 28 * 28)
    max_samples: int = field(default=-1, metadata={"help": "Truncate every annotation file (debugging)"})
    shuffle: bool = field(default=True)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    auto_resume: bool = field(
        default=True, metadata={"help": "Resume from the latest checkpoint-* in output_dir if present"}
    )
