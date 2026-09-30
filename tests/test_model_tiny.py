"""Tiny random DynThinker model on CPU (stub geometry encoder): forward/backward, generation, save/load."""

import copy

import pytest
import torch

from conftest import build_inputs, tiny_config


def make_model(modeling, depart=False, seed=0):
    torch.manual_seed(seed)
    model = modeling.Qwen3VLForConditionalGenerationWithVGGT(tiny_config(depart)).eval()
    with torch.no_grad():
        for layer in model.model.language_model.layers:
            if layer.cross_flag:
                layer.cross_attn.gate.fill_(0.7)
    return model


def test_sdf_layers_and_frozen_modules(stub_geometry):
    model = make_model(stub_geometry)
    names = [n for n, _ in model.named_parameters()]
    assert sorted({int(n.split(".")[3]) for n in names if ".cross_attn." in n}) == [3, 4, 5]
    assert any(n.startswith("model.geometry_merger.") for n in names)


@pytest.mark.parametrize("num_frames,depart", [(4, False), (9, True)])
def test_forward_backward(stub_geometry, num_frames, depart):
    model = make_model(stub_geometry, depart)
    model.train()
    inputs = build_inputs(num_frames, depart)
    out = model(**inputs, labels=inputs["input_ids"].clone())
    assert torch.isfinite(out.loss)
    out.loss.backward()
    sdf_grads = [p.grad for n, p in model.named_parameters() if ".cross_attn." in n]
    assert all(g is not None for g in sdf_grads)
    assert any(g.abs().sum() > 0 for g in sdf_grads)


def test_geometry_changes_outputs(stub_geometry):
    model = make_model(stub_geometry)
    inputs = build_inputs(4)
    no_geo = {k: v for k, v in inputs.items() if k != "geometry_encoder_inputs"}
    with torch.no_grad():
        assert not torch.allclose(model(**copy.deepcopy(inputs)).logits, model(**no_geo).logits)


def test_generate(stub_geometry):
    model = make_model(stub_geometry)
    inputs = build_inputs(4)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=4, do_sample=False, pad_token_id=0, eos_token_id=None)
    assert out.shape == (1, inputs["input_ids"].shape[1] + 4)


def test_save_and_reload_round_trip(stub_geometry, tmp_path):
    model = make_model(stub_geometry)
    inputs = build_inputs(4)
    with torch.no_grad():
        reference = model(**copy.deepcopy(inputs)).logits
    model.save_pretrained(tmp_path / "ckpt")
    reloaded = stub_geometry.Qwen3VLForConditionalGenerationWithVGGT.from_pretrained(
        tmp_path / "ckpt", attn_implementation="eager"
    ).eval()
    assert reloaded.config.geo_inject_version == "v32"
    with torch.no_grad():
        assert torch.equal(reloaded(**copy.deepcopy(inputs)).logits, reference)

    # VDPM weights are loaded explicitly; the research-era keyword fails instead of being absorbed.
    with pytest.raises(TypeError, match="load_vdpm_weights"):
        stub_geometry.Qwen3VLForConditionalGenerationWithVGGT.from_pretrained(tmp_path / "ckpt", geometry_encoder_path="x.pt")
    loaded = []
    reloaded.model.geometry_encoder.load_model = loaded.append
    reloaded.load_vdpm_weights("vdpm.pt")
    assert loaded == ["vdpm.pt"]


@pytest.mark.parametrize(
    "key,value",
    [("geo_inject_version", "v6"), ("feature_fusion_method", "add"), ("predict_next_frame", True)],
)
def test_unsupported_research_options_fail_loudly(stub_geometry, key, value):
    cfg = tiny_config()
    setattr(cfg, key, value)
    setattr(cfg.text_config, key, value)
    with pytest.raises(ValueError):
        stub_geometry.Qwen3VLForConditionalGenerationWithVGGT(cfg)
