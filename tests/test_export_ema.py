"""`dynthinker-export-ema` reproduces the end-of-training D-EMA application from a saved checkpoint."""

import json

import pytest
import torch
from safetensors.torch import load_file

from conftest import tiny_config
from dynthinker.train.dema import DecoupledEMACallback
from dynthinker.train.export_ema import export_ema_checkpoint, main


class State:
    def __init__(self, step):
        self.global_step = step
        self.is_world_process_zero = True


class Args:
    def __init__(self, output_dir):
        self.output_dir = str(output_dir)


def _load_all(directory):
    tensors = {}
    for path in sorted(directory.glob("*.safetensors")):
        tensors.update(load_file(str(path)))
    return tensors


@pytest.mark.parametrize("shard", ["20MB", "200MB"])
def test_export_matches_final_application(stub_geometry, tmp_path, shard):
    torch.manual_seed(0)
    model = stub_geometry.Qwen3VLForConditionalGenerationWithVGGT(tiny_config()).to(torch.bfloat16)
    for name, p in model.named_parameters():
        p.requires_grad = "language_model" in name or name.startswith("lm_head")

    cb = DecoupledEMACallback(mlp_decay=0.9, other_decay=0.8, start_step=1, update_interval=1, checkpoint_save=True)
    out_dir = tmp_path / "run"
    cb.on_train_begin(Args(out_dir), State(0), None, model=model)
    gen = torch.Generator().manual_seed(0)
    for step in range(1, 4):
        with torch.no_grad():
            for p in model.parameters():
                if p.requires_grad:
                    p.add_(torch.randn(p.shape, generator=gen).to(p.dtype) * 0.05)
        cb.on_step_end(Args(out_dir), State(step), None, model=model)

    ckpt = out_dir / "checkpoint-3"
    model.save_pretrained(ckpt, max_shard_size=shard)  # online weights, as in a Trainer checkpoint
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json"):
        (ckpt / name).write_bytes(b"resume-only")
    (ckpt / "global_step3").mkdir()
    cb.on_save(Args(out_dir), State(3), None, model=model)

    summary = export_ema_checkpoint(str(ckpt), str(tmp_path / "exported"))
    assert summary["replaced_tensors"] == len(cb.param_names["mlp"]) + len(cb.param_names["other"])
    exported = _load_all(tmp_path / "exported")
    leftovers = {"optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json", "global_step3",
                 "ema_state.safetensors", "ema_state.json"}
    assert not leftovers & {p.name for p in (tmp_path / "exported").iterdir()}
    assert (tmp_path / "exported" / "config.json").exists()
    marker = json.loads((tmp_path / "exported" / "dema_export.json").read_text())
    assert marker["resumable"] is False and marker["groups"] == ["mlp", "other"]

    cb.maybe_apply_to_model(model)
    expected = model.state_dict()
    assert set(exported) <= set(expected)
    for key, tensor in exported.items():
        assert torch.equal(tensor, expected[key]), key
    for name in cb.param_names["new_module"]:
        assert torch.equal(exported[name], _load_all(ckpt)[name])  # online weights kept


def test_export_refuses_existing_output_and_missing_state(tmp_path):
    (tmp_path / "ckpt").mkdir()
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "file").write_text("x")
    with pytest.raises(FileExistsError):
        export_ema_checkpoint(str(tmp_path / "ckpt"), str(tmp_path / "out"))
    with pytest.raises(FileNotFoundError):
        export_ema_checkpoint(str(tmp_path / "ckpt"), str(tmp_path / "new"))


def test_cli_help(capsys):
    with pytest.raises(SystemExit) as info:
        main(["--help"])
    assert info.value.code == 0
    assert "--checkpoint" in capsys.readouterr().out


def test_training_refuses_to_resume_from_an_export(tmp_path):
    from types import SimpleNamespace

    from dynthinker.train.train import find_resume_checkpoint

    for name in ("checkpoint-2", "checkpoint-10", "checkpoint-12-dema", "checkpoint-20"):
        (tmp_path / name).mkdir()
    (tmp_path / "checkpoint-20" / "dema_export.json").write_text("{}")
    args = SimpleNamespace(resume_from_checkpoint=None, auto_resume=True, output_dir=str(tmp_path))
    assert find_resume_checkpoint(args, None) == str(tmp_path / "checkpoint-10")
    args.resume_from_checkpoint = str(tmp_path / "checkpoint-20")
    with pytest.raises(ValueError, match="cannot be resumed"):
        find_resume_checkpoint(args, None)


def test_final_output_dir_is_marked_non_resumable(tmp_path):
    from types import SimpleNamespace

    from dynthinker.train.train import find_resume_checkpoint, write_final_marker

    write_final_marker(str(tmp_path), ["mlp", "other"])
    marker = json.loads((tmp_path / "dema_export.json").read_text())
    assert marker["resumable"] is False and marker["groups"] == ["mlp", "other"]
    args = SimpleNamespace(resume_from_checkpoint=str(tmp_path), auto_resume=True, output_dir=str(tmp_path))
    with pytest.raises(ValueError, match="cannot be resumed"):
        find_resume_checkpoint(args, None)
