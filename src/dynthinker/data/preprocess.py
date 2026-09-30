# Copyright (c) 2026 The DynThinker Authors. Licensed under the MIT License.
"""Per-frame inputs shared by training: Qwen-VL pixel patches plus the VDPM frame tensor."""

from .image_utils import vggt_crop_preprocess


def prepare_image_inputs(image, image_processor):
    """Preprocess one frame for both streams.

    The VDPM input is the VGGT crop-mode tensor (width 518, height a multiple of 14). The Qwen-VL input
    is the top-left part of that tensor whose sides are multiples of ``patch_size * merge_size``,
    passed to the image processor as values already in [0, 1] (``do_rescale=False``).

    Returns ``pixel_values``, ``image_grid_thw`` (the [t, h, w] grid of this frame) and
    ``geometry_encoder_inputs`` ([3, H, W]).
    """
    frame = vggt_crop_preprocess(image)
    unit = image_processor.patch_size * image_processor.merge_size
    rows = frame.shape[1] // unit * unit
    cols = frame.shape[2] // unit * unit
    qwen_input = frame[None, :, :rows, :cols].clone()
    processed = image_processor(qwen_input, return_tensors="pt", do_rescale=False)
    return {
        "pixel_values": processed["pixel_values"],
        "image_grid_thw": processed["image_grid_thw"][0],
        "geometry_encoder_inputs": frame,
    }
