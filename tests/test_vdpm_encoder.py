"""Real VDPM architecture (random weights) on CPU. Slow (~1.2B parameters, ~26 GB peak RAM): set DYNTHINKER_SLOW_TESTS=1."""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("DYNTHINKER_SLOW_TESTS") != "1", reason="set DYNTHINKER_SLOW_TESTS=1")


def test_vdpm_dual_stream_shapes_and_state_dict_roundtrip(tmp_path):
    from dynthinker.model.geometry_encoders import create_geometry_encoder

    torch.manual_seed(0)
    encoder = create_geometry_encoder(
        "vdpm",
        vdpm_use_decoder_features=True,
        vdpm_temporal_mode="diag",
        vdpm_decoder_layer_offset=-2,
        vdpm_concat_encoder_decoder_features=True,
    )
    assert encoder.target_layer_idx == 17
    assert not any(p.requires_grad for p in encoder.parameters())
    frames = torch.rand(3, 3, 28, 42)
    features, camera, extra = encoder.encode(frames)
    assert features.shape == (6, 2 * 3, 2048)  # [encoder frames; decoder frames], 2 x 3 patches
    assert camera.shape == (3, 1, 2048) and extra == {}
    assert not torch.allclose(features[:3], features[3:])

    # load_model() accepts a plain state dict (weights_only=True) such as the public VDPM model.pt.
    path = tmp_path / "vdpm.pt"
    torch.save(encoder.vdpm.state_dict(), path)
    other = create_geometry_encoder("vdpm", vdpm_concat_encoder_decoder_features=True)
    other.load_model(str(path))
    features2, _, _ = other.encode(frames)
    assert torch.equal(features, features2)
