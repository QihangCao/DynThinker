# Copyright (c) 2026 The DynThinker Authors. Licensed under the MIT License.
# Wraps V-DPM (https://github.com/eldar/vdpm); see src/dynthinker/model/vdpm/ for its license.
"""Frozen VDPM dual-stream feature extractor.

For an input sequence of S frames the wrapper returns ``2 * S`` feature maps:
the VDPM encoder (VGGT-style aggregator) tokens of all frames, followed by the
time-conditioned decoder tokens of all frames. The decoder is conditioned
"diagonally": frame t is decoded at time t in a single decoder pass. Both
streams are taken from the same intermediate layer (``vdpm_decoder_layer_offset``,
default -2, i.e. aggregator block 17 of 24).

Every input frame is decoded. Subset decoding with temporal interpolation is not provided by this
wrapper.
"""

import contextlib
import logging
from typing import List

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class VDPMEncoder(nn.Module):
    """VDPM geometry encoder wrapper (encoder + decoder feature streams)."""

    patch_size = 14
    feature_dim = 2048

    def __init__(
        self,
        freeze_encoder: bool = True,
        vdpm_use_decoder_features: bool = True,
        vdpm_temporal_mode: str = "diag",
        vdpm_decoder_layer_offset: int = -2,
        vdpm_compute_prediction: bool = False,
        vdpm_concat_encoder_decoder_features: bool = False,
    ):
        super().__init__()
        from ..vdpm.model import VDPM

        if vdpm_temporal_mode != "diag":
            raise ValueError(f"Only vdpm_temporal_mode='diag' is supported in this release, got {vdpm_temporal_mode!r}")
        if not vdpm_concat_encoder_decoder_features:
            raise ValueError(
                "DynThinker requires vdpm_concat_encoder_decoder_features=True "
                "(SDF consumes both the encoder and the decoder stream)"
            )
        if vdpm_compute_prediction:
            raise ValueError("vdpm_compute_prediction is not supported in this release")

        self.freeze_encoder = bool(freeze_encoder)
        self.use_decoder_features = bool(vdpm_use_decoder_features)
        self.temporal_feature_mode = vdpm_temporal_mode
        self.decoder_layer_offset = int(vdpm_decoder_layer_offset)
        self.concat_encoder_decoder_features = True
        self.vdpm = VDPM(skip_decoder=False)

        self.target_layer_idx = self._resolve_target_layer_idx(self.decoder_layer_offset)
        decoder_indices = self.vdpm.decoder.intermediate_layer_idx
        if self.target_layer_idx not in decoder_indices:
            raise ValueError(
                f"Target layer index {self.target_layer_idx} is not produced by decoder. "
                f"Available decoder layers: {decoder_indices}"
            )
        self.target_decoder_pos = decoder_indices.index(self.target_layer_idx)
        self._apply_freeze()

    def _apply_freeze(self) -> None:
        if self.freeze_encoder:
            self.vdpm.requires_grad_(False)

    def _resolve_target_layer_idx(self, layer_offset: int) -> int:
        total_layers = len(self.vdpm.point_head.intermediate_layer_idx)
        layer_idx = total_layers + layer_offset if layer_offset < 0 else layer_offset
        if layer_idx < 0 or layer_idx >= total_layers:
            raise ValueError(
                f"vdpm_decoder_layer_offset resolves to invalid index {layer_idx} for {total_layers} decoder outputs"
            )
        return self.vdpm.point_head.intermediate_layer_idx[layer_idx]

    def _run_decoder_diag(
        self,
        images: torch.Tensor,
        aggregated_tokens_list: List[torch.Tensor],
        patch_start_idx: int,
    ) -> torch.Tensor:
        # Condition frame t on time t for every frame in one decoder pass. Returns [S, N, C].
        num_frames = images.shape[0]
        cond_view_idxs = torch.arange(num_frames, dtype=torch.int64, device=images.device).unsqueeze(0)
        decoded_tokens = self.vdpm.decoder(images[None], aggregated_tokens_list, patch_start_idx, cond_view_idxs)
        return decoded_tokens[self.target_decoder_pos][0, :, patch_start_idx:]

    @staticmethod
    def _autocast_for(images: torch.Tensor):
        if images.is_cuda:
            major = torch.cuda.get_device_capability(images.device)[0]
            dtype = torch.bfloat16 if major >= 8 else torch.float16
            return torch.autocast("cuda", dtype=dtype)
        # CPU execution (tests / debugging) runs in the input precision.
        return contextlib.nullcontext()

    @torch.no_grad()
    def encode(self, images: torch.Tensor, **kwargs):
        """images: [S, 3, H, W] in [0, 1]. Returns (features [2S, N, C], camera_token [S, 1, C], {})."""
        self.vdpm.eval()
        with self._autocast_for(images):
            aggregated_tokens_list, patch_start_idx = self.vdpm.aggregator(images[None])
            camera_token = aggregated_tokens_list[self.target_layer_idx][0, :, 0:1]
            encoder_features = aggregated_tokens_list[self.target_layer_idx][0, :, patch_start_idx:]
            decoder_features = self._run_decoder_diag(images, aggregated_tokens_list, patch_start_idx)
            features = torch.cat([encoder_features, decoder_features], dim=0)
        return features, camera_token, {}

    def get_feature_dim(self) -> int:
        return self.feature_dim

    def forward(self, images: torch.Tensor):
        return self.encode(images)

    def load_model(self, model_path: str) -> None:
        """Load pretrained VDPM weights (a plain state dict, e.g. model.pt from the V-DPM release)."""
        from ..vdpm.model import VDPM

        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        self.vdpm = VDPM(skip_decoder=False)
        self.vdpm.load_state_dict(state_dict)
        logger.info("Loaded VDPM weights from %s", model_path)
        self._apply_freeze()


def create_geometry_encoder(encoder_type: str = "vdpm", **kwargs) -> VDPMEncoder:
    """Build the geometry encoder named by the config (only "vdpm" exists in this release)."""
    if encoder_type != "vdpm":
        raise ValueError(f"Only geometry_encoder_type='vdpm' is supported in this release, got {encoder_type!r}")
    return VDPMEncoder(**kwargs)
