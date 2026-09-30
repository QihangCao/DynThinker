"""Dataset registry, frame sampling, preprocessing and collation (no tokenizer download needed)."""

import json
import os
import types

import numpy as np
import pytest
import torch

from conftest import image_processor, synthetic_frames
from dynthinker.data.dataset import DataCollatorForSupervisedDataset, LazySupervisedDataset
from dynthinker.data.preprocess import prepare_image_inputs
from dynthinker.data.registry import data_list, load_registry, parse_sampling_rate

EXAMPLES = os.path.join(os.path.dirname(__file__), os.pardir, "examples")


def test_registry_resolution(tmp_path, monkeypatch):
    monkeypatch.setenv("TOY_ROOT", str(tmp_path / "data"))
    (tmp_path / "registry.yaml").write_text(
        "spar:\n  annotation_path: ${TOY_ROOT}/spar.json\n  data_path: media\n  tag: 3d\n"
        "hound:\n  annotation_path: hound.jsonl\n"
    )
    registry = load_registry(str(tmp_path / "registry.yaml"))
    assert registry["spar"]["annotation_path"] == str(tmp_path / "data" / "spar.json")
    assert registry["spar"]["data_path"] == str(tmp_path / "media")
    assert registry["hound"]["tag"] == "2d"
    configs = data_list(["spar%25", "hound"], registry)
    assert [c["sampling_rate"] for c in configs] == [0.25, 1.0]
    assert parse_sampling_rate("x%100") == 1.0
    with pytest.raises(ValueError):
        data_list(["unknown"], registry)


def test_registry_rejects_bad_entries(tmp_path, monkeypatch):
    monkeypatch.delenv("UNSET_ROOT_FOR_TEST", raising=False)
    (tmp_path / "a.json").write_text(json.dumps({"x": {"annotation_path": "a.json", "tag": "4d"}}))
    with pytest.raises(ValueError):
        load_registry(str(tmp_path / "a.json"))
    (tmp_path / "b.json").write_text(json.dumps({"x": {"annotation_path": "${UNSET_ROOT_FOR_TEST}/a.json"}}))
    with pytest.raises(ValueError):
        load_registry(str(tmp_path / "b.json"))


def test_example_files_are_valid():
    registry = load_registry(os.path.join(EXAMPLES, "datasets.example.json"))
    assert set(registry) == {"toy_video", "toy_multiview"}
    for name in registry:
        with open(registry[name]["annotation_path"], "r", encoding="utf-8") as handle:
            records = json.load(handle)
        for record in records:
            assert record["conversations"][0]["from"] == "human"
            assert record["conversations"][1]["from"] == "gpt"


def test_frame_indices():
    ds = LazySupervisedDataset.__new__(LazySupervisedDataset)
    ds.data_args = types.SimpleNamespace(base_interval=2, video_min_frames=4, video_max_frames=8)
    assert list(ds.frame_indices(6, 1)) == [0, 1, 3, 5]  # 6 s / 2 s = 3 frames -> clamped up to 4
    assert list(ds.frame_indices(10, 1)) == [0, 2, 4, 6, 9]  # 10 s / 2 s = 5 frames
    assert len(ds.frame_indices(300, 1)) == 8  # clamped down to video_max_frames
    ds.data_args.video_max_frames = 32
    assert len(ds.frame_indices(3000, 30)) == 32


def test_length_estimate_accepts_content_key():
    ds = LazySupervisedDataset.__new__(LazySupervisedDataset)
    ds.data_args = types.SimpleNamespace(video_max_frames=8)
    ds.depart_smi_token, ds.smi_image_num, ds.smi_downsample_rate = True, 8, 2
    turns = [{"from": "human", "content": "<image> two words"}, {"from": "gpt", "value": "A"}]
    assert ds._estimate_length({"images": ["a.jpg"] * 2, "conversations": turns}) == 2 * 252 + 4
    assert ds._estimate_length({"video": "v.mp4", "conversations": turns}) == 8 * 252 + 4
    ds.data_args.video_max_frames = 16
    assert ds._estimate_length({"video": "v.mp4", "conversations": turns}) == 16 * 252 // 4 + 4


def test_spar_info_records_are_rejected(tmp_path):
    record = {"images": ["a.jpg"], "spar_info": "{}", "conversations": [{"from": "human", "value": "<image>"}]}
    (tmp_path / "a.json").write_text(json.dumps([record]))
    (tmp_path / "reg.json").write_text(json.dumps({"s": {"annotation_path": "a.json", "tag": "3d"}}))
    args = types.SimpleNamespace(dataset_use="s", dataset_config=str(tmp_path / "reg.json"), max_samples=-1,
                                 model_type="qwen3vl", image_processor=image_processor())
    with pytest.raises(ValueError, match="spar_info"):
        LazySupervisedDataset(None, args)


def test_prepare_image_inputs_shapes():
    proc = image_processor()
    ret = prepare_image_inputs(synthetic_frames(1)[0], proc)
    geo = ret["geometry_encoder_inputs"]
    assert geo.shape == (3, 294, 518)  # width 518, height a multiple of 14
    assert 0.0 <= float(geo.min()) and float(geo.max()) <= 1.0
    t, h, w = ret["image_grid_thw"].tolist()
    assert (t, h, w) == (1, 18, 32)  # 288 x 512 crop, patch 16
    assert ret["pixel_values"].shape == (h * w, 3 * 2 * 16 * 16)


def test_collator_pads_and_stacks():
    tokenizer = types.SimpleNamespace(pad_token_id=0, model_max_length=64)
    proc = image_processor()
    instances = []
    for n in (2, 3):
        rets = [prepare_image_inputs(f, proc) for f in synthetic_frames(n)]
        length = 5 + n
        instances.append(
            dict(
                input_ids=torch.arange(1, length + 1),
                labels=torch.full((length,), -100),
                position_ids=torch.arange(length).view(1, 1, -1).expand(3, 1, -1),
                pixel_values=[r["pixel_values"] for r in rets],
                image_grid_thw=[r["image_grid_thw"] for r in rets],
                geometry_encoder_inputs=[r["geometry_encoder_inputs"] for r in rets],
                tag="3d",
            )
        )
    batch = DataCollatorForSupervisedDataset(tokenizer)(instances)
    assert batch["input_ids"].shape == (2, 8)
    assert batch["attention_mask"].sum().item() == 7 + 8
    assert batch["position_ids"].shape == (3, 2, 8)
    assert batch["image_grid_thw"].shape == (5, 3)
    assert [g.shape[0] for g in batch["geometry_encoder_inputs"]] == [2, 3]
    assert batch["tag"] == "3d"
    instances[1]["tag"] = "2d"
    with pytest.raises(ValueError):
        DataCollatorForSupervisedDataset(tokenizer)(instances)


@pytest.mark.skipif(not os.environ.get("DYNTHINKER_TEST_TOKENIZER"), reason="set DYNTHINKER_TEST_TOKENIZER=<Qwen3-VL dir>")
def test_dataset_end_to_end_with_tokenizer(tmp_path):
    from transformers import AutoProcessor, AutoTokenizer

    path = os.environ["DYNTHINKER_TEST_TOKENIZER"]
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False, padding_side="right")
    media = tmp_path / "frames"
    media.mkdir()
    for i, frame in enumerate(synthetic_frames(6)):
        frame.save(media / f"{i:03d}.jpg")
    (tmp_path / "train.json").write_text(json.dumps([{
        "id": "0", "video": "frames",
        "conversations": [{"from": "human", "value": "<video>\nQ?"}, {"from": "gpt", "value": "A"}],
    }]))
    (tmp_path / "reg.json").write_text(json.dumps({"toy": {"annotation_path": "train.json", "data_path": ".", "tag": "3d"}}))
    args = types.SimpleNamespace(
        dataset_use="toy", dataset_config=str(tmp_path / "reg.json"), video_max_frames=8, video_min_frames=4,
        base_interval=2, max_pixels=576 * 28 * 28, min_pixels=16 * 28 * 28, max_samples=-1, shuffle=False,
        model_type="qwen3vl", use_geometry_encoder=True,
        image_processor=AutoProcessor.from_pretrained(path).image_processor,
    )
    item = LazySupervisedDataset(tokenizer, args)[0]
    assert len(item["geometry_encoder_inputs"]) == 4
    supervised = item["labels"][item["labels"] != -100]
    assert tokenizer.decode(supervised).strip().startswith("A")
