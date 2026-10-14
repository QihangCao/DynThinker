# Copyright (c) 2026 The DynThinker Authors. Licensed under the MIT License.
"""Spatial-Dynamic Fusion (SDF).

This is the interaction module selected by ``geo_inject_version="v1"`` in DynThinker
checkpoints. Its parameter names match the released checkpoint structure (``cross_attn``)
so trained weights load without key remapping.
"""

import torch
import torch.nn as nn


class SpatialDynamicFusion(nn.Module):
    """Fuse VDPM encoder (spatial) and decoder (dynamic) features into image-token states.

    Given image-token hidden states ``H`` and projected geometry tokens laid out as
    ``[encoder tokens ; decoder tokens]`` along the sequence axis, it returns the residual
    update ``tanh(g) * (W_os(LN(F_enc) * S_s) + W_od(LN(F_dec) * S_d))`` with
    ``S_s = MLP_s(LN(H))`` and ``S_d = MLP_d(LN(H))`` (sigmoid-bounded per-token scores).
    The caller adds the update to ``H``. ``g`` is initialised to zero, so the module is an
    identity mapping at initialisation.
    """

    def __init__(self, config, layer_idx=None, **kwargs):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.encoder_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.decoder_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

        # W_os (spatial / encoder stream) and W_od (dynamic / decoder stream).
        self.os_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.ot_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        # Importance gating: bottleneck MLPs predicting one score per visual token.
        self.spatial_importance_net = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 4),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 4, 1),
            nn.Sigmoid(),
        )
        self.temporal_importance_net = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 4),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 4, 1),
            nn.Sigmoid(),
        )

        # Global residual scale, applied as tanh(gate).
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, semantic_hidden, vggt_features, **kwargs):
        """
        semantic_hidden: [1, N_img_tokens, C] image-token hidden states at this layer.
        vggt_features:   [1, 2 * N_img_tokens, C] projected encoder tokens followed by decoder tokens.
        Returns the residual update with the same shape as ``semantic_hidden``.
        """
        encoder_features, decoder_features = vggt_features.chunk(2, dim=-2)

        semantic_hidden = self.input_layernorm(semantic_hidden)
        encoder_features = self.encoder_layernorm(encoder_features)
        decoder_features = self.decoder_layernorm(decoder_features)

        spatial_importance = self.spatial_importance_net(semantic_hidden)
        temporal_importance = self.temporal_importance_net(semantic_hidden)

        output = self.os_proj(encoder_features * spatial_importance) + self.ot_proj(
            decoder_features * temporal_importance
        )
        return torch.tanh(self.gate) * output


# Release-facing class alias; checkpoint tensors remain registered under `cross_attn`.
QwenVGGTInteractionv1 = SpatialDynamicFusion

SDF_INJECT_VERSION = "v1"


def is_sdf_layer(layer_idx: int, num_hidden_layers: int) -> bool:
    """SDF is deployed on the middle LLM layers: quarter < layer_idx < num_layers - quarter.

    This reproduces the placement rule used by the DynThinker SDF implementation (``v1``).
    """
    quarter_num = num_hidden_layers // 4
    return (layer_idx > quarter_num) and (layer_idx < (num_hidden_layers - quarter_num))
