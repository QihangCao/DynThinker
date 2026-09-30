"""Spatial-Dynamic Fusion: formula, identity at initialisation and layer placement."""

import types

import torch

from dynthinker.model.sdf import SpatialDynamicFusion, is_sdf_layer


def make_sdf(hidden=32, seed=0):
    torch.manual_seed(seed)
    return SpatialDynamicFusion(types.SimpleNamespace(hidden_size=hidden, rms_norm_eps=1e-6))


def test_identity_at_initialisation():
    sdf = make_sdf()
    assert sdf.gate.item() == 0.0
    h = torch.randn(1, 10, 32)
    geo = torch.randn(1, 20, 32)
    assert torch.count_nonzero(sdf(h, geo)) == 0


def test_matches_paper_formula():
    sdf = make_sdf()
    with torch.no_grad():
        sdf.gate.fill_(0.8)
    h = torch.randn(1, 6, 32)
    enc = torch.randn(1, 6, 32)
    dec = torch.randn(1, 6, 32)
    out = sdf(h, torch.cat([enc, dec], dim=1))

    ln = torch.nn.functional.layer_norm
    h_n = ln(h, (32,), sdf.input_layernorm.weight, sdf.input_layernorm.bias, 1e-6)
    s_s = torch.sigmoid(sdf.spatial_importance_net[2](torch.relu(sdf.spatial_importance_net[0](h_n))))
    s_d = torch.sigmoid(sdf.temporal_importance_net[2](torch.relu(sdf.temporal_importance_net[0](h_n))))
    f_enc = ln(enc, (32,), sdf.encoder_layernorm.weight, sdf.encoder_layernorm.bias, 1e-6)
    f_dec = ln(dec, (32,), sdf.decoder_layernorm.weight, sdf.decoder_layernorm.bias, 1e-6)
    fused = f_enc * s_s @ sdf.os_proj.weight.T + f_dec * s_d @ sdf.ot_proj.weight.T
    expected = torch.tanh(torch.tensor(0.8)) * fused
    assert s_s.shape == (1, 6, 1) and s_d.shape == (1, 6, 1)
    assert torch.allclose(out, expected, atol=1e-6)


def test_streams_are_split_along_the_token_axis():
    sdf = make_sdf()
    with torch.no_grad():
        sdf.gate.fill_(1.0)
        sdf.ot_proj.weight.zero_()  # remove the dynamic stream
    h = torch.randn(1, 4, 32)
    enc = torch.randn(1, 4, 32)
    out_a = sdf(h, torch.cat([enc, torch.randn(1, 4, 32)], dim=1))
    out_b = sdf(h, torch.cat([enc, torch.randn(1, 4, 32)], dim=1))
    assert torch.allclose(out_a, out_b)  # depends on the first half (encoder) only


def test_parameter_names_are_checkpoint_compatible():
    names = {n for n, _ in make_sdf().named_parameters()}
    assert names == {
        "input_layernorm.weight", "input_layernorm.bias",
        "encoder_layernorm.weight", "encoder_layernorm.bias",
        "decoder_layernorm.weight", "decoder_layernorm.bias",
        "os_proj.weight", "ot_proj.weight",
        "spatial_importance_net.0.weight", "spatial_importance_net.0.bias",
        "spatial_importance_net.2.weight", "spatial_importance_net.2.bias",
        "temporal_importance_net.0.weight", "temporal_importance_net.0.bias",
        "temporal_importance_net.2.weight", "temporal_importance_net.2.bias",
        "gate",
    }


def test_layer_placement_middle_half():
    # Qwen3-VL-8B has 36 layers: quarter = 9 -> SDF on layers 10..26.
    layers = [i for i in range(36) if is_sdf_layer(i, 36)]
    assert layers == list(range(10, 27))
    assert [i for i in range(8) if is_sdf_layer(i, 8)] == [3, 4, 5]
