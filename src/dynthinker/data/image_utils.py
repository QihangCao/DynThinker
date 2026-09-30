# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the VGGT License found in
# LICENSES/VGGT-LICENSE.txt at the root of this repository.
#
# Adapted from `load_and_preprocess_images` (mode="crop") in vggt/utils/load_fn.py
# (https://github.com/facebookresearch/vggt). Modified by the DynThinker authors: one frame per call,
# given as a PIL image or a path; the multi-image padding and "pad" mode are not needed and removed.
"""VGGT/V-DPM input preprocessing for a single frame."""

import torch
from PIL import Image
from torchvision import transforms as TF

TARGET_SIZE = 518


def vggt_crop_preprocess(image) -> torch.Tensor:
    """Return a [3, H, W] float tensor in [0, 1] with W = 518 and H a multiple of 14 (at most 518).

    As in VGGT's "crop" mode: transparent pixels become white, the frame is resized (bicubic) to width 518
    keeping the aspect ratio with the height rounded to a multiple of 14, and a taller result is
    center-cropped to 518 rows.
    """
    img = Image.open(image) if not isinstance(image, Image.Image) else image
    if img.mode == "RGBA":
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img)
    img = img.convert("RGB")

    width, height = img.size
    new_width = TARGET_SIZE
    new_height = round(height * (new_width / width) / 14) * 14
    img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
    tensor = TF.ToTensor()(img)

    if new_height > TARGET_SIZE:
        start_y = (new_height - TARGET_SIZE) // 2
        tensor = tensor[:, start_y : start_y + TARGET_SIZE, :]
    return tensor
