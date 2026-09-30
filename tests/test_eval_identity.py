# Copyright (c) 2026 The DynThinker Authors. Licensed under the MIT License.
"""Checkpoint identity must cover trained middle layers without reading unrelated files."""
import json

import pytest

from dynthinker.eval import run


def _middle_change(path):
    with path.open("r+b") as handle:
        handle.seek(3 << 19)
        handle.write(b"updated trainable middle parameters")


def test_unsharded_weights_middle_change(tmp_path):
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"0" * (3 << 20))
    (tmp_path / "config.json").write_text("{}")
    before = run._model_fingerprint(str(tmp_path))
    _middle_change(weights)
    assert run._model_fingerprint(str(tmp_path)) != before


def test_only_model_assets_are_read(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"synthetic weights")
    (tmp_path / "unrelated.json").write_text('{"synthetic_note":true}')
    visited = []
    real_hash = run._sha256_file

    def observe(path):
        visited.append(str(path))
        assert not str(path).endswith("unrelated.json")
        return real_hash(path)

    monkeypatch.setattr(run, "_sha256_file", observe)
    assert set(run._model_fingerprint(str(tmp_path))) == {"config.json", "model.safetensors"}
    assert len(visited) == 2


def test_non_json_tokenizer_assets_are_identified(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"synthetic weights")
    merges = tmp_path / "merges.txt"
    merges.write_text("a b\n")
    before = run._model_fingerprint(str(tmp_path))
    merges.write_text("b c\n")
    assert run._model_fingerprint(str(tmp_path)) != before


def test_indexed_shard_middle_change(tmp_path):
    shard = tmp_path / "part.safetensors"
    shard.write_bytes(b"0" * (3 << 20))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.layer.weight": shard.name}})
    )
    before = run._model_fingerprint(str(tmp_path))
    _middle_change(shard)
    assert run._model_fingerprint(str(tmp_path)) != before


@pytest.mark.parametrize("shard", ["../outside.bin", "/outside.bin", "unrelated.json"])
def test_invalid_index_paths_are_rejected(tmp_path, shard):
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": shard}})
    )
    with pytest.raises(ValueError):
        run._model_fingerprint(str(tmp_path))


def test_external_vdpm_middle_change(tmp_path):
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"0" * (3 << 20))
    before = run._weight_fingerprint(str(weights))
    _middle_change(weights)
    assert run._weight_fingerprint(str(weights)) != before


def test_legacy_chat_template_json_is_identified(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"synthetic weights")
    template = tmp_path / "chat_template.json"
    template.write_text(json.dumps({"chat_template": "original prompt"}))
    before = run._model_fingerprint(str(tmp_path))
    template.write_text(json.dumps({"chat_template": "changed prompt"}))
    assert "chat_template.json" in before
    assert run._model_fingerprint(str(tmp_path)) != before


@pytest.mark.parametrize("kind", ["missing", "file", "empty_directory"])
def test_invalid_model_location_cannot_have_an_empty_identity(tmp_path, kind):
    model = tmp_path / "checkpoint"
    if kind == "file":
        model.write_bytes(b"not a checkpoint directory")
    elif kind == "empty_directory":
        model.mkdir()
        (model / "config.json").write_text("{}")
    with pytest.raises(ValueError):
        run._model_fingerprint(str(model))
