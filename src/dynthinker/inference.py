# Copyright (c) 2026 The DynThinker Authors. Licensed under the MIT License.
# Resizing helper adapted from GeoThinker (MIT; see LICENSES/MIT-GeoThinker.txt).
# Evaluation crop and processor input assembly are reimplemented from the required tensor
# behavior of VG-LLM/GeoThinker. See THIRD_PARTY_NOTICES.md for provenance.
"""Load a DynThinker checkpoint and answer questions about a frame sequence."""

import logging
import os
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F
from PIL import Image

from .data.image_utils import vggt_crop_preprocess

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = "You are a helpful assistant."
_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def adaptive_resize_pad(tensor: torch.Tensor, target_shape, mode: str = "bilinear") -> torch.Tensor:
    """Downscale (never upscale) a [C, H, W] tensor to fit ``target_shape`` and zero-pad bottom/right."""
    curr_h, curr_w = tensor.shape[-2:]
    target_h, target_w = target_shape
    scale = min(1.0, target_h / curr_h, target_w / curr_w)
    if scale < 1.0:
        new_h, new_w = int(curr_h * scale), int(curr_w * scale)
        tensor = F.interpolate(tensor.unsqueeze(0), size=(new_h, new_w), mode=mode, align_corners=False).squeeze(0)
        curr_h, curr_w = new_h, new_w
    return F.pad(tensor, (0, target_w - curr_w, 0, target_h - curr_h), mode="constant", value=0)


def prepare_frames(frames: Sequence[Image.Image], patch_size: int, merge_size: int):
    """Build normalized Qwen images and a separate [frames, 3, H, W] geometry batch.

    Geometry frames use the VGGT crop and the first frame's canvas. For evaluation, the Qwen crop
    retains complete groups of patches AND any final sub-patch pixels. This historical convention
    differs from training, which keeps only complete pixel blocks.
    """
    if patch_size < 1 or merge_size < 1:
        raise ValueError("Patch and merge sizes must be positive")
    normalized = tuple(vggt_crop_preprocess(frame.convert("RGB")) for frame in frames)
    if not normalized:
        raise ValueError("At least one frame is required")
    canvas = normalized[0].shape[-2:]
    aligned = [frame if frame.shape[-2:] == canvas else adaptive_resize_pad(frame, canvas).clamp(0, 1)
               for frame in normalized]
    group_pixels = patch_size * merge_size
    images = []
    for frame in aligned:
        rows, columns = (int(size) // group_pixels * group_pixels + int(size) % patch_size
                         for size in frame.shape[-2:])
        images.append(frame[..., :rows, :columns])
    return images, torch.stack(aligned, dim=0)


# Settings that every released DynThinker checkpoint records in its config.json.
_REQUIRED_CONFIG = {
    "use_geometry_encoder": True,
    "geo_cross_attn": True,
    "geo_inject_version": "v32",
    "vdpm_use_decoder_features": True,
    "vdpm_concat_encoder_decoder_features": True,
}


def check_dynthinker_config(config) -> None:
    """Reject checkpoints that are not DynThinker (e.g. a plain Qwen3-VL checkpoint) instead of silently
    running them without VDPM/SDF."""
    if getattr(config, "model_type", None) != "qwen3_vl":
        raise ValueError(f"Not a Qwen3-VL checkpoint (model_type={getattr(config, 'model_type', None)!r})")
    wrong = {
        key: getattr(config, key, None)
        for key, value in _REQUIRED_CONFIG.items()
        if getattr(config, key, None) != value
    }
    if wrong:
        raise ValueError(
            "This checkpoint is not a DynThinker (VDPM + SDF v32) checkpoint; config has "
            + ", ".join(f"{key}={value!r}" for key, value in wrong.items())
            + ". A plain Qwen3-VL checkpoint cannot be used with this pipeline."
        )


def check_loaded_weights(missing_keys, geometry_encoder_loaded: bool) -> None:
    """Fail if SDF / geometry-merger weights, or VDPM weights without --geometry-encoder-path, were missing."""
    missing = list(missing_keys or [])
    fusion = [key for key in missing if "cross_attn" in key or "geometry_merger" in key]
    if fusion:
        raise ValueError(f"Checkpoint lacks {len(fusion)} SDF/geometry-merger weights (e.g. {fusion[0]!r})")
    vdpm = [key for key in missing if "geometry_encoder" in key]
    if vdpm and not geometry_encoder_loaded:
        raise ValueError(
            f"Checkpoint lacks {len(vdpm)} VDPM weights; pass --geometry-encoder-path "
            "(or set DYNTHINKER_VDPM_PATH) to the VDPM model.pt"
        )
    other = [key for key in missing if key not in fusion and key not in vdpm]
    if other:
        raise ValueError(f"Checkpoint lacks {len(other)} model weights (e.g. {other[0]!r})")


class DynThinkerPipeline:
    """Minimal inference wrapper around ``Qwen3VLForConditionalGenerationWithVGGT``."""

    def __init__(
        self,
        model_path: str,
        geometry_encoder_path: Optional[str] = None,
        device: str = "cuda",
        dtype: str = "bfloat16",
        attn_implementation: str = "sdpa",
        min_pixels: int = 12544,
        max_pixels: int = 451584,
    ):
        from transformers import AutoConfig, AutoTokenizer

        from .model.modeling_qwen3_vl import Qwen3VLForConditionalGenerationWithVGGT
        from .model.processing_qwen3_vl import Qwen3VLProcessor

        config = AutoConfig.from_pretrained(model_path)
        check_dynthinker_config(config)
        setattr(config, "training", False)

        geometry_encoder_path = geometry_encoder_path or os.environ.get("DYNTHINKER_VDPM_PATH") or None
        if getattr(config, "exclude_geometry_encoder_on_save", False) and not geometry_encoder_path:
            raise ValueError(
                "This checkpoint was saved without VDPM weights; pass --geometry-encoder-path "
                "(or set DYNTHINKER_VDPM_PATH) to the VDPM model.pt"
            )

        self.device = torch.device(device)
        self.dtype = _DTYPES[dtype]
        model, loading_info = Qwen3VLForConditionalGenerationWithVGGT.from_pretrained(
            model_path,
            config=config,
            torch_dtype=self.dtype,
            attn_implementation=attn_implementation,
            output_loading_info=True,
        )
        if geometry_encoder_path is not None:
            model.load_vdpm_weights(geometry_encoder_path)
        check_loaded_weights(loading_info.get("missing_keys"), geometry_encoder_loaded=bool(geometry_encoder_path))
        # VDPM weights loaded from a separate file start in float32 on CPU; move/cast everything.
        self.model = model.to(self.device).to(self.dtype).eval()

        self.processor = Qwen3VLProcessor.from_pretrained(
            model_path,
            max_pixels=max_pixels,
            min_pixels=min_pixels,
            padding_side="left",
            depart_smi_token=bool(getattr(config, "depart_smi_token", False)),
            smi_image_num=getattr(config, "smi_image_num", 8),
            smi_downsample_rate=getattr(config, "smi_downsample_rate", 2),
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")

    def build_inputs(self, frames: Sequence[Image.Image], question: str, add_system_prompt: bool = True):
        if not frames:
            raise ValueError("At least one frame is required")
        messages = []
        if add_system_prompt:
            messages.append({"role": "system", "content": SYSTEM_PROMPT})
        messages.append(
            {"role": "user", "content": [{"type": "image", "image": f} for f in frames] + [{"type": "text", "text": question}]}
        )
        text = self.processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=True)

        image_processor = self.processor.image_processor
        qwen_images, geometry_inputs = prepare_frames(frames, image_processor.patch_size, image_processor.merge_size)
        encoding = {
            "images": qwen_images,
            "text": text,
            "return_tensors": "pt",
            "padding": True,
            "do_rescale": False,  # frames already contain normalized [0, 1] pixels
        }
        inputs = self.processor(**encoding)
        inputs["geometry_encoder_inputs"] = geometry_inputs.unsqueeze(0)
        return inputs.to(self.device)

    @torch.no_grad()
    def generate(
        self,
        frames: Sequence[Image.Image],
        question: str,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        add_system_prompt: bool = True,
    ) -> str:
        # Geometry inputs stay in float32 (VDPM runs under autocast on CUDA), as in the evaluation adapter.
        inputs = self.build_inputs(frames, question, add_system_prompt=add_system_prompt)
        output = self.model.generate(
            **inputs,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=top_p,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
        trimmed = output[:, inputs["input_ids"].shape[1] :]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def load_frames(video: Optional[str] = None, images: Optional[List[str]] = None, num_frames: int = 32) -> List[Image.Image]:
    """Frames from a video file (uniformly sampled) or from an ordered list of image files."""
    if bool(video) == bool(images):
        raise ValueError("Provide exactly one of a video or a list of images")
    if video:
        from .video import read_video_frames

        return read_video_frames(video, num_frames=num_frames)
    frames = []
    for path in images:
        with Image.open(path) as image:
            frames.append(image.convert("RGB").copy())
    return frames
