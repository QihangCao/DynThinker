# Copyright (c) 2026 The DynThinker Authors. Licensed under the MIT License.
# `Qwen2RMSNorm` is imported from HuggingFace Transformers (Apache License 2.0).
"""Projection of VDPM patch features onto the LLM visual-token grid."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm


def _projector(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, out_dim))


class GeometryFeatureMerger(nn.Module):
    """Merge each ``m x m`` block of geometry patches into one LLM-width token (m = spatial merge size).

    Input ``x`` is ``[n_maps, rows, cols, context_dim]``; each patch vector is RMS-normalised (``ln_q``),
    the ``m x m`` block is flattened in (row, column, channel) order and projected by ``mlp``.
    ``camera_tokens`` (``[n_maps, 1, context_dim]``) are projected by ``camera_mlp``. Returns
    ``(tokens [n_maps, rows // m, cols // m, output_dim], projected camera tokens)``.

    With ``depart_smi_token`` and more than ``smi_image_num`` maps, the patch grid is first bilinearly
    shrunk by ``smi_downsample_rate``. The released model passes encoder and decoder maps together, so
    ``n_maps`` is twice the number of frames while ``smi_image_num`` keeps its default of 8; this matches
    the research implementation that produced the checkpoints and is intentionally left unchanged.

    Parameter names (``ln_q``, ``mlp.{0,2}``, ``camera_mlp.{0,2}``) match the released checkpoints.
    """

    def __init__(
        self,
        output_dim: int,
        hidden_dim: int,
        context_dim: int,
        spatial_merge_size: int = 2,
        merger_type: str = "mlp",
        depart_smi_token: bool = False,
        smi_image_num: int = 8,
        smi_downsample_rate: int = 2,
    ):
        super().__init__()
        if merger_type != "mlp":
            raise ValueError(f"Only geometry_merger_type='mlp' is supported in this release, got {merger_type!r}")
        self.merge_size = spatial_merge_size
        self.depart_smi_token = bool(depart_smi_token)
        self.smi_image_num = smi_image_num
        self.smi_downsample_rate = smi_downsample_rate

        self.ln_q = Qwen2RMSNorm(context_dim, eps=1e-6)
        self.mlp = _projector(context_dim * spatial_merge_size**2, hidden_dim, output_dim)
        self.camera_mlp = _projector(context_dim, hidden_dim, output_dim)

    def _shrink(self, x: torch.Tensor) -> torch.Tensor:
        rows, cols = x.shape[1] // self.smi_downsample_rate, x.shape[2] // self.smi_downsample_rate
        grid = F.interpolate(x.permute(0, 3, 1, 2), size=(rows, cols), mode="bilinear", align_corners=False)
        return grid.permute(0, 2, 3, 1)

    def forward(self, x: torch.Tensor, camera_tokens: torch.Tensor):
        if self.depart_smi_token and x.shape[0] > self.smi_image_num:
            x = self._shrink(x)
        m = self.merge_size
        rows, cols = x.shape[1] // m, x.shape[2] // m
        blocks = rearrange(x[:, : rows * m, : cols * m], "n (r a) (c b) d -> n r c (a b) d", a=m, b=m)
        blocks = self.ln_q(blocks.contiguous()).flatten(-2)
        return self.mlp(blocks), self.camera_mlp(camera_tokens)
