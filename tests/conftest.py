"""Shared helpers for the CPU test-suite (tiny random models; no weights or network needed)."""

import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))

IMAGE_TOKEN, VIDEO_TOKEN, VSTART, VEND = 151655, 151656, 151652, 151653


class StubGeometryEncoder(nn.Module):
    """Deterministic stand-in for VDPM with the same interface: patch 14, 2048-d, [encoder; decoder] maps."""

    patch_size = 14

    def __init__(self, *args, **kwargs):
        super().__init__()
        gen = torch.Generator().manual_seed(1234)
        self.register_buffer("proj", torch.randn(3, 2048, generator=gen), persistent=False)
        self.register_buffer("cam", torch.randn(3, 2048, generator=gen), persistent=False)

    def get_feature_dim(self):
        return 2048

    def encode(self, images, **kwargs):
        s = images.shape[0]
        tokens = F.avg_pool2d(images.float(), 14).flatten(2).transpose(1, 2)
        enc = torch.tanh(tokens @ self.proj)
        dec = torch.sin(enc * 3.0 + torch.arange(s, dtype=torch.float32).view(s, 1, 1))
        camera = (images.float().mean(dim=(2, 3)) @ self.cam).unsqueeze(1)
        return torch.cat([enc, dec], dim=0), camera, {}


@pytest.fixture
def stub_geometry(monkeypatch):
    import dynthinker.model.modeling_qwen3_vl as modeling

    monkeypatch.setattr(modeling, "create_geometry_encoder", lambda *a, **k: StubGeometryEncoder())
    return modeling


def tiny_config(depart_smi_token=False, num_layers=8, use_geometry_encoder=True):
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig

    cfg = Qwen3VLConfig(
        text_config=dict(
            vocab_size=151680,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=num_layers,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=4096,
            rope_scaling={"rope_type": "default", "mrope_section": [2, 3, 3], "mrope_interleaved": True},
            rms_norm_eps=1e-6,
            tie_word_embeddings=False,
        ),
        vision_config=dict(
            depth=2,
            hidden_size=32,
            intermediate_size=64,
            num_heads=2,
            out_hidden_size=64,
            patch_size=16,
            spatial_merge_size=2,
            temporal_patch_size=2,
            deepstack_visual_indexes=[0],
            num_position_embeddings=2304,
        ),
        image_token_id=IMAGE_TOKEN,
        video_token_id=VIDEO_TOKEN,
        vision_start_token_id=VSTART,
        vision_end_token_id=VEND,
        tie_word_embeddings=False,
    )
    overrides = dict(
        use_geometry_encoder=use_geometry_encoder,
        geometry_encoder_type="vdpm",
        feature_fusion_method="zero",
        cam_merger_type="zero",
        geometry_merger_type="mlp",
        geometry_merger_hidden_dim=32,
        geo_cross_attn=use_geometry_encoder,
        geo_inject_version="v32",
        vdpm_use_decoder_features=True,
        vdpm_temporal_mode="diag",
        vdpm_decoder_layer_offset=-2,
        vdpm_compute_prediction=False,
        vdpm_concat_encoder_decoder_features=True,
        depart_smi_token=depart_smi_token,
        smi_image_num=8,
        smi_downsample_rate=2,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
        setattr(cfg.text_config, key, value)
    for sub in (cfg, cfg.text_config, cfg.vision_config):
        sub._attn_implementation = "eager"
    return cfg


def image_processor():
    from transformers import Qwen2VLImageProcessor

    return Qwen2VLImageProcessor(
        patch_size=16,
        temporal_patch_size=2,
        merge_size=2,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        min_pixels=16 * 32 * 32,
        max_pixels=4096 * 32 * 32,
    )


def synthetic_frames(n, size=(640, 360), seed=0):
    rng = np.random.default_rng(seed)
    return [
        Image.fromarray(rng.integers(0, 255, size=(size[1] // 8, size[0] // 8, 3), dtype=np.uint8)).resize(
            size, Image.BILINEAR
        )
        for _ in range(n)
    ]


def build_inputs(num_frames, depart_smi_token=False):
    """Token ids / pixels / VDPM inputs for `num_frames` synthetic frames (mirrors the data pipeline)."""
    from dynthinker.data.preprocess import prepare_image_inputs

    proc = image_processor()
    pixel_values, grids, geo = [], [], []
    for frame in synthetic_frames(num_frames):
        ret = prepare_image_inputs(frame, proc)
        pixel_values.append(ret["pixel_values"])
        grids.append(ret["image_grid_thw"])
        geo.append(ret["geometry_encoder_inputs"])
    merge = 2 * (2 if depart_smi_token and num_frames > 8 else 1)
    ids = [11, 12, 13]
    for g in grids:
        ids += [VSTART] + [IMAGE_TOKEN] * int(g[0] * (g[1] // merge) * (g[2] // merge)) + [VEND]
    ids += [21, 22, 23, 24]
    input_ids = torch.tensor([ids])
    return dict(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        pixel_values=torch.cat(pixel_values, 0),
        image_grid_thw=torch.stack(grids, 0),
        geometry_encoder_inputs=[torch.stack(geo)],
    )
