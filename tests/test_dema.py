"""D-EMA: parameter groups, late-MLP boundary, update period, closed form, export semantics, checkpoint round trip."""

import json

import pytest
import torch
import torch.nn as nn

from dynthinker.train.dema import (
    EMA_META_FILENAME,
    EMA_STATE_FILENAME,
    DecoupledEMACallback,
    expected_update_count,
    group_parameter_names,
    is_update_step,
)


class Layer(nn.Module):
    def __init__(self, with_sdf=False):
        super().__init__()
        self.self_attn = nn.Linear(4, 4)
        self.mlp = nn.Sequential(nn.Linear(4, 4))
        if with_sdf:
            self.cross_attn = nn.Linear(4, 4)


class LanguageModel(nn.Module):
    def __init__(self, num_layers=6):
        super().__init__()
        self.embed_tokens = nn.Embedding(10, 4)
        self.layers = nn.ModuleList([Layer(with_sdf=i in (2, 3)) for i in range(num_layers)])


class Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.geometry_merger = nn.Linear(4, 4)
        self.visual = nn.Linear(4, 4)
        self.language_model = LanguageModel()


class ToyModel(nn.Module):
    """Parameter names follow Qwen3-VL: model.language_model.layers.{i}.mlp/self_attn/cross_attn, lm_head."""

    def __init__(self):
        super().__init__()
        self.model = Inner()
        self.lm_head = nn.Linear(4, 10)
        for p in self.model.visual.parameters():
            p.requires_grad = False
        for p in self.model.geometry_merger.parameters():
            p.requires_grad = False


class State:
    def __init__(self, step):
        self.global_step = step
        self.is_world_process_zero = True


class Args:
    def __init__(self, output_dir):
        self.output_dir = str(output_dir)


def make_callback(**overrides):
    kwargs = dict(
        mlp_decay=0.9,
        other_decay=0.5,
        new_module_decay=-1.0,
        mlp_layer_threshold=4,
        start_step=3,
        update_interval=2,
        checkpoint_save=True,
    )
    kwargs.update(overrides)
    return DecoupledEMACallback(**kwargs)


def perturb(model, gen):
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.add_(torch.randn(p.shape, generator=gen))


def test_groups_and_late_mlp_boundary():
    groups = group_parameter_names(ToyModel().named_parameters(), mlp_layer_threshold=4)
    mlp_layers = sorted({int(n.split(".")[3]) for n in groups["mlp"]})
    assert mlp_layers == [4, 5]  # research-code rule: 0-based layer index >= threshold
    assert "model.language_model.layers.3.mlp.0.weight" in groups["other"]
    assert "model.language_model.layers.4.self_attn.weight" in groups["other"]
    assert all(".cross_attn." in n for n in groups["new_module"]) and groups["new_module"]
    assert "lm_head.weight" in groups["other"] and "model.language_model.embed_tokens.weight" in groups["other"]
    everything = groups["mlp"] + groups["other"] + groups["new_module"]
    assert not any("geometry_merger" in n or "visual" in n for n in everything)  # frozen -> excluded
    assert len(everything) == len(set(everything))


def test_expected_update_count_is_inclusive():
    assert [expected_update_count(s, 3, 2) for s in range(0, 10)] == [0, 0, 0, 1, 1, 2, 2, 3, 3, 4]


def test_update_period_and_closed_form(tmp_path):
    model = ToyModel()
    cb = make_callback()
    cb.on_train_begin(Args(tmp_path), State(0), None, model=model)
    names = dict(model.named_parameters())
    initial = {n: names[n].detach().clone() for n in cb.param_names["mlp"] + cb.param_names["other"]}
    history = []
    gen = torch.Generator().manual_seed(0)
    for step in range(1, 12):
        perturb(model, gen)
        cb.on_step_end(Args(tmp_path), State(step), None, model=model)
        if step >= 3 and (step - 3) % 2 == 0:
            history.append({n: p.detach().clone() for n, p in model.named_parameters()})
    assert cb.num_updates == len(history) == 5 and cb.last_ema_step == 11

    for group, rho in (("mlp", 0.9), ("other", 0.5)):
        for name in cb.param_names[group]:
            n_updates = len(history)
            expected = rho**n_updates * initial[name]
            for t, snapshot in enumerate(history, start=1):
                expected = expected + (1 - rho) * rho ** (n_updates - t) * snapshot[name]
            assert torch.allclose(cb.shadows[group][name], expected, atol=1e-5), name


def test_new_modules_keep_online_weights_at_export(tmp_path):
    model = ToyModel()
    cb = make_callback()
    cb.on_train_begin(Args(tmp_path), State(0), None, model=model)
    gen = torch.Generator().manual_seed(1)
    for step in range(1, 8):
        perturb(model, gen)
        cb.on_step_end(Args(tmp_path), State(step), None, model=model)
    online = {n: p.detach().clone() for n, p in model.named_parameters()}
    applied = cb.maybe_apply_to_model(model)
    assert applied == len(cb.param_names["mlp"]) + len(cb.param_names["other"])
    params = dict(model.named_parameters())
    for name in cb.param_names["new_module"]:
        assert torch.equal(params[name], online[name])  # rho_new = 0 at export
    for group in ("mlp", "other"):
        for name in cb.param_names[group]:
            assert torch.equal(params[name], cb.shadows[group][name].to(params[name].dtype))
            assert not torch.equal(params[name], online[name])
    frozen = "model.geometry_merger.weight"
    assert torch.equal(params[frozen], online[frozen])


def test_checkpoint_round_trip_and_strict_checks(tmp_path):
    model = ToyModel()
    cb = make_callback()
    args = Args(tmp_path)
    cb.on_train_begin(args, State(0), None, model=model)
    gen = torch.Generator().manual_seed(2)
    for step in range(1, 8):
        perturb(model, gen)
        cb.on_step_end(args, State(step), None, model=model)
    cb.on_save(args, State(7), None, model=model)
    ckpt = tmp_path / "checkpoint-7"
    assert (ckpt / EMA_STATE_FILENAME).is_file() and (ckpt / EMA_META_FILENAME).is_file()
    meta = json.loads((ckpt / EMA_META_FILENAME).read_text())
    assert meta["num_updates"] == 3 and meta["global_step"] == 7
    assert set(meta["groups"]) == {"mlp", "other", "new_module"}

    resumed = make_callback()
    resumed.set_resume_checkpoint(ckpt)
    resumed.on_train_begin(args, State(7), None, model=model)
    for group, shadow in cb.shadows.items():
        for name, value in shadow.items():
            assert torch.equal(resumed.shadows[group][name], value)
    assert resumed.num_updates == 3

    # Continuing after resume matches an uninterrupted run.
    perturb(model, gen)
    cb.on_step_end(args, State(9), None, model=model)
    resumed.on_step_end(args, State(9), None, model=model)
    for group, shadow in cb.shadows.items():
        for name, value in shadow.items():
            assert torch.equal(resumed.shadows[group][name], value)

    with pytest.raises(RuntimeError, match="configuration mismatch"):
        other = make_callback(mlp_decay=0.95)
        other.set_resume_checkpoint(ckpt)
        other.on_train_begin(args, State(7), None, model=model)
    with pytest.raises(RuntimeError, match="step mismatch"):
        other = make_callback()
        other.set_resume_checkpoint(ckpt)
        other.on_train_begin(args, State(8), None, model=model)
    with pytest.raises(RuntimeError, match="missing"):
        other = make_callback()
        other.set_resume_checkpoint(tmp_path / "does-not-exist")
        other.on_train_begin(args, State(7), None, model=model)
    with pytest.raises(RuntimeError, match="inconsistent"):
        cb.save_checkpoint_state(str(tmp_path / "bad"), 100)


def test_checkpoint_dtype_bfloat16(tmp_path):
    from safetensors import safe_open

    model = ToyModel()
    cb = make_callback(checkpoint_dtype="bfloat16")
    cb.on_train_begin(Args(tmp_path), State(0), None, model=model)
    cb.save_checkpoint_state(str(tmp_path / "ckpt"), 0)
    with safe_open(str(tmp_path / "ckpt" / EMA_STATE_FILENAME), framework="pt") as reader:
        assert all(reader.get_tensor(k).dtype == torch.bfloat16 for k in reader.keys())


def test_rank_zero_only_non_owner_is_inert(tmp_path):
    model = ToyModel()
    cb = make_callback(rank_zero_only=True)
    state = State(0)
    state.is_world_process_zero = False
    cb.on_train_begin(Args(tmp_path), state, None, model=model)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    cb.on_step_end(Args(tmp_path), State(3), None, model=model)
    assert cb.num_updates == 0 and cb.maybe_apply_to_model(model) == 0
    assert all(torch.equal(p, before[n]) for n, p in model.named_parameters())


def test_invalid_configuration_and_naming_guard(tmp_path):
    with pytest.raises(ValueError):
        DecoupledEMACallback(mlp_decay=-1.0)
    with pytest.raises(ValueError):
        DecoupledEMACallback(mlp_decay=0.9, checkpoint_dtype="int8")

    class Unprefixed(nn.Module):  # decoder layers not under `language_model`
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Layer() for _ in range(6)])

    cb = make_callback()
    with pytest.raises(RuntimeError, match="naming"):
        cb.on_train_begin(Args(tmp_path), State(0), None, model=Unprefixed())


def test_expected_count_matches_simulated_schedule():
    for start in (0, 1, 3, 150):
        for interval in (1, 2, 10):
            fired = 0
            for step in range(1, 400):
                fired += is_update_step(step, start, interval)
                assert expected_update_count(step, start, interval) == fired, (start, interval, step)
    # Schedules with start_step >= 1 keep the research code's count ((step - start) // interval + 1).
    assert all(expected_update_count(s, 150, 10) == (s - 150) // 10 + 1 for s in range(150, 1000))


@pytest.mark.parametrize("interval", [1, 3])
def test_start_step_zero_checkpoint_and_resume(tmp_path, interval):
    model = ToyModel()
    cb = make_callback(start_step=0, update_interval=interval)
    args = Args(tmp_path)
    cb.on_train_begin(args, State(0), None, model=model)
    gen = torch.Generator().manual_seed(3)
    for step in range(1, 7):
        perturb(model, gen)
        cb.on_step_end(args, State(step), None, model=model)
        assert cb.num_updates == step // interval
        cb.on_save(args, State(step), None, model=model)  # would raise on an inconsistent count

    resumed = make_callback(start_step=0, update_interval=interval)
    resumed.set_resume_checkpoint(tmp_path / "checkpoint-6")
    resumed.on_train_begin(args, State(6), None, model=model)
    assert resumed.num_updates == 6 // interval
    for group, shadow in cb.shadows.items():
        for name, value in shadow.items():
            assert torch.equal(resumed.shadows[group][name], value)


def test_resume_without_ema_state_fails_unless_opted_in(tmp_path):
    args = Args(tmp_path)
    for kwargs in (dict(checkpoint_save=False), dict(checkpoint_save=True)):  # no state / no checkpoint given
        with pytest.raises(RuntimeError, match="no EMA state is restored"):
            make_callback(**kwargs).on_train_begin(args, State(8), None, model=ToyModel())
    make_callback(checkpoint_save=False).on_train_begin(args, State(0), None, model=ToyModel())  # fresh run is fine

    model = ToyModel()
    cb = make_callback(checkpoint_save=False, reinit_on_resume=True)
    cb.on_train_begin(args, State(8), None, model=model)
    assert cb.num_updates == expected_update_count(8, 3, 2)
    shadows = {name: p.detach() for name, p in model.named_parameters() if p.requires_grad}
    assert all(torch.equal(cb.shadows[g][n], shadows[n]) for g in cb.shadows for n in cb.shadows[g])


def _tiny_text_config(**extra):
    return dict(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=6,
                num_attention_heads=2, num_key_value_heads=1, **extra)


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen2_5_vl"])
def test_groups_on_real_transformers_parameter_names(backbone):
    """The sensitive group is non-empty for both HF Qwen-VL naming schemes (transformers 4.57)."""
    import transformers

    if backbone == "qwen3_vl":
        config = transformers.Qwen3VLConfig(
            text_config=_tiny_text_config(head_dim=8, rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]}),
            vision_config=dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2, out_hidden_size=16,
                               deepstack_visual_indexes=[0]),
        )
        model = transformers.Qwen3VLForConditionalGeneration(config)
    else:
        config = transformers.Qwen2_5_VLConfig(
            text_config=_tiny_text_config(rope_scaling={"type": "mrope", "mrope_section": [1, 1, 2]}),
            vision_config=dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2, out_hidden_size=16),
        )
        model = transformers.Qwen2_5_VLForConditionalGeneration(config)
    groups = group_parameter_names(model.named_parameters(), mlp_layer_threshold=4)
    layers = sorted({int(n.split("layers.")[1].split(".")[0]) for n in groups["mlp"]})
    assert layers == [4, 5]
    assert all(n.startswith("model.language_model.layers.") and ".mlp." in n for n in groups["mlp"])
    assert "model.language_model.layers.3.mlp.up_proj.weight" in groups["other"]
    assert not groups["new_module"]
