"""Every Trainer save carries the processor files, so the D-EMA export of an intermediate checkpoint loads as is.

The default tests build a tiny byte-level tokenizer and processor (no downloads). The last test runs
`dynthinker-train` end to end with the real Qwen3-VL tokenizer/processor files and is skipped unless
DYNTHINKER_TEST_TOKENIZER points to a Qwen3-VL directory (only its tokenizer/processor files are read).
"""

import json
import os
import shutil
import subprocess
import sys

import pytest
import torch
from safetensors.torch import load_file

from conftest import StubGeometryEncoder, build_inputs, synthetic_frames, tiny_config
from dynthinker.train.export_ema import PROCESSOR_FILENAMES
from dynthinker.train.export_ema import main as export_main

HERE = os.path.dirname(os.path.abspath(__file__))
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
# The processor and tokenizer templates differ on purpose: the export must render prompts with the
# processor template, as the base model's processor does.
PROCESSOR_TEMPLATE = (
    "{% for message in messages %}<|im_start|>{{ message['role'] }}\n"
    "{% if message['content'] is string %}{{ message['content'] }}{% else %}{% for item in message['content'] %}"
    "{% if item['type'] == 'image' %}<|vision_start|><|image_pad|><|vision_end|>{% else %}{{ item['text'] }}{% endif %}"
    "{% endfor %}{% endif %}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)
TOKENIZER_TEMPLATE = "{% for message in messages %}[{{ message['role'] }}] {{ message['content'] }}\n{% endfor %}"
TRAIN_MAX_PIXELS, TRAIN_MIN_PIXELS = 256 * 32 * 32, 8 * 32 * 32


class LoadableStubGeometryEncoder(StubGeometryEncoder):
    def load_model(self, path):
        assert os.path.isfile(path)

    def encode(self, images, **kwargs):  # the training entry point loads the model in bfloat16
        self.proj, self.cam = self.proj.float(), self.cam.float()
        return super().encode(images.float(), **kwargs)


@pytest.fixture
def stub_vdpm(monkeypatch, tmp_path):
    import dynthinker.model.modeling_qwen3_vl as modeling

    monkeypatch.setattr(modeling, "create_geometry_encoder", lambda *a, **k: LoadableStubGeometryEncoder())
    path = tmp_path / "stub-vdpm.pt"
    path.write_bytes(b"")
    return str(path)


def _tiny_base(directory):
    """Tiny random Qwen3-VL with a byte-level Qwen2 tokenizer and a Qwen3-VL processor (legacy template file)."""
    import transformers
    from transformers.models.qwen2.tokenization_qwen2 import bytes_to_unicode

    torch.manual_seed(0)
    transformers.Qwen3VLForConditionalGeneration(tiny_config(use_geometry_encoder=False)).save_pretrained(directory)
    vocab = {char: index for index, char in enumerate(bytes_to_unicode().values())}
    (directory / "vocab.json").write_text(json.dumps(vocab))
    (directory / "merges.txt").write_text("#version: 0.2\n")
    tokenizer = transformers.Qwen2TokenizerFast(
        vocab_file=str(directory / "vocab.json"),
        merges_file=str(directory / "merges.txt"),
        unk_token=None,
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        additional_special_tokens=["<|im_start|>", "<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>"],
    )
    tokenizer.chat_template = TOKENIZER_TEMPLATE
    tokenizer.save_pretrained(directory)
    transformers.Qwen2VLImageProcessorFast(
        patch_size=16, temporal_patch_size=2, merge_size=2, image_mean=[0.5] * 3, image_std=[0.5] * 3,
        size={"shortest_edge": 65536, "longest_edge": 16777216},
    ).save_pretrained(directory)
    transformers.Qwen3VLVideoProcessor(size={"shortest_edge": 4096, "longest_edge": 25165824}).save_pretrained(directory)
    (directory / "chat_template.json").write_text(json.dumps({"chat_template": PROCESSOR_TEMPLATE}, indent=4))
    return directory


class _Batches(torch.utils.data.Dataset):
    def __init__(self, count):
        sample = build_inputs(2)
        sample["labels"] = sample["input_ids"].clone()
        self.samples = [sample] * count

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def _run(base, output_dir, vdpm_path, geo_aware, max_steps, resume_from=None):
    """Train as `dynthinker.train.train.train` does, on pre-tokenized tiny batches."""
    from transformers import AutoTokenizer, TrainerCallback

    from dynthinker.train.arguments import DataArguments, ModelArguments, TrainingArguments
    from dynthinker.train.dema import DecoupledEMACallback
    from dynthinker.train.train import GeoAwareTrainer, build_model, set_trainable
    from dynthinker.train.trainer import DynThinkerTrainer

    model_args = ModelArguments(
        model_name_or_path=str(base), attn_implementation="eager", tune_mm_llm=True,
        use_geometry_encoder=True, geometry_encoder_path=vdpm_path, geo_cross_attn=True,
        exclude_geometry_encoder_on_save=geo_aware, ema_enable=True, ema_start_step=1, ema_update_interval=1,
        ema_mlp_decay=0.9, ema_other_decay=0.8, ema_checkpoint_save=True,
    )
    data_args = DataArguments(max_pixels=TRAIN_MAX_PIXELS, min_pixels=TRAIN_MIN_PIXELS)
    training_args = TrainingArguments(
        output_dir=str(output_dir), use_cpu=True, max_steps=max_steps, save_strategy="steps", save_steps=1,
        per_device_train_batch_size=1, learning_rate=1e-3, lr_scheduler_type="constant", logging_steps=1,
        report_to="none", seed=0, dataloader_num_workers=0, remove_unused_columns=False,
    )
    model, backbone, processor = build_model(model_args, training_args, data_args, "eager")
    assert data_args.image_processor is processor.image_processor
    # What the training dataset does to the image processor (see LazySupervisedDataset).
    data_args.image_processor.max_pixels = data_args.max_pixels
    data_args.image_processor.min_pixels = data_args.min_pixels
    data_args.image_processor.size["longest_edge"] = data_args.max_pixels
    data_args.image_processor.size["shortest_edge"] = data_args.min_pixels
    model = model.float()
    set_trainable(model_args, model, backbone)
    if geo_aware:
        model._exclude_geometry_encoder_on_save = True

    ema = DecoupledEMACallback(
        mlp_decay=0.9, other_decay=0.8, start_step=1, update_interval=1, mlp_layer_threshold=4, checkpoint_save=True
    )
    captured = {}

    class Capture(TrainerCallback):
        def on_train_begin(self, args, state, control, model=None, **kwargs):
            captured["step"] = state.global_step
            captured["weights"] = {k: v.detach().clone() for k, v in model.state_dict().items()}
            captured["shadows"] = {f"{g}:{n}": t.clone() for g, s in ema.shadows.items() for n, t in s.items()}

    if resume_from is not None:
        ema.set_resume_checkpoint(resume_from)
    trainer_cls = GeoAwareTrainer if geo_aware else DynThinkerTrainer
    trainer = trainer_cls(
        model=model,
        processing_class=AutoTokenizer.from_pretrained(str(base), model_max_length=4096, padding_side="right"),
        args=training_args,
        callbacks=[ema, Capture()],
        train_dataset=_Batches(4),
        data_collator=lambda features: features[0],
        processor=processor,
    )
    trainer.train(resume_from_checkpoint=resume_from)
    return trainer, processor, captured


def _assert_processor_matches(directory, processor):
    """Load `directory` as DynThinkerPipeline does and compare with the processor used for training."""
    from transformers import AutoProcessor, AutoTokenizer

    from dynthinker.model.processing_qwen3_vl import Qwen3VLProcessor

    loaded = Qwen3VLProcessor.from_pretrained(str(directory))
    assert loaded.chat_template == processor.chat_template
    messages = [[{"role": "system", "content": "sys"},
                 {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Where?"}]}]]
    rendered = loaded.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    assert rendered == processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image, expected_image = loaded.image_processor, processor.image_processor
    assert type(image) is type(expected_image)
    for key in ("max_pixels", "min_pixels", "size", "patch_size", "merge_size", "temporal_patch_size",
                "image_mean", "image_std"):
        assert getattr(image, key) == getattr(expected_image, key), key
    assert loaded.video_processor.to_dict() == processor.video_processor.to_dict()
    assert AutoProcessor.from_pretrained(str(directory)).chat_template == processor.chat_template
    assert loaded.tokenizer.get_vocab() == processor.tokenizer.get_vocab()
    return loaded, AutoTokenizer.from_pretrained(str(directory))


@pytest.mark.parametrize("geo_aware", [False, True], ids=["DynThinkerTrainer", "GeoAwareTrainer"])
def test_intermediate_export_loads_processor_without_extra_files(stub_vdpm, tmp_path, geo_aware):
    from dynthinker.train.train import save_final_model

    base = _tiny_base(tmp_path / "base")
    run = tmp_path / "run"
    trainer, processor, _ = _run(base, run, stub_vdpm, geo_aware, max_steps=2)
    assert processor.chat_template == PROCESSOR_TEMPLATE
    assert processor.image_processor.max_pixels == TRAIN_MAX_PIXELS

    for step in (1, 2):
        checkpoint = run / f"checkpoint-{step}"
        names = {p.name for p in checkpoint.iterdir()}
        assert set(PROCESSOR_FILENAMES) <= names, sorted(names)
        assert {"model.safetensors", "ema_state.safetensors", "ema_state.json", "optimizer.pt"} <= names
        assert set(TOKENIZER_FILES) <= names

    exported = tmp_path / "exported-step1"
    assert export_main(["--checkpoint", str(run / "checkpoint-1"), "--output", str(exported)]) == 0
    assert not {"optimizer.pt", "ema_state.safetensors", "trainer_state.json"} & {p.name for p in exported.iterdir()}
    assert json.loads((exported / "dema_export.json").read_text())["resumable"] is False
    _, tokenizer = _assert_processor_matches(exported, processor)
    assert tokenizer.chat_template == TOKENIZER_TEMPLATE  # the tokenizer keeps its own template
    assert tokenizer.model_max_length == 4096

    # The export holds the D-EMA weights, the checkpoint keeps the online weights for resuming.
    online = load_file(str(run / "checkpoint-1" / "model.safetensors"))
    averaged = load_file(str(exported / "model.safetensors"))
    assert sum(not torch.equal(online[k], averaged[k]) for k in online) > 0

    save_final_model(trainer, str(run), exclude_geometry_encoder_on_save=geo_aware)
    for name in PROCESSOR_FILENAMES:
        assert (run / name).read_bytes() == (run / "checkpoint-2" / name).read_bytes(), name
    _assert_processor_matches(run, processor)


def test_checkpoint_with_processor_files_still_resumes_online_and_ema_state(stub_vdpm, tmp_path):
    base = _tiny_base(tmp_path / "base")
    _run(base, tmp_path / "run", stub_vdpm, geo_aware=False, max_steps=2)
    checkpoint = tmp_path / "run" / "checkpoint-1"

    resumed_run = tmp_path / "resumed"
    shutil.copytree(checkpoint, resumed_run / "checkpoint-1")
    _, processor, captured = _run(base, resumed_run, stub_vdpm, geo_aware=False, max_steps=2,
                                  resume_from=str(resumed_run / "checkpoint-1"))
    assert captured["step"] == 1
    online = load_file(str(checkpoint / "model.safetensors"))
    for key, value in online.items():
        assert torch.equal(captured["weights"][key].to(value.dtype), value), key
    shadows = load_file(str(checkpoint / "ema_state.safetensors"))
    assert shadows and set(shadows) == set(captured["shadows"])
    for key, value in shadows.items():
        assert torch.equal(captured["shadows"][key], value), key

    # Resuming reproduces the uninterrupted run and keeps writing processor files.
    for name in ("model.safetensors", "ema_state.safetensors"):
        expected = load_file(str(tmp_path / "run" / "checkpoint-2" / name))
        actual = load_file(str(resumed_run / "checkpoint-2" / name))
        assert set(actual) == set(expected)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=1e-6)
    for name in PROCESSOR_FILENAMES:
        assert (resumed_run / "checkpoint-2" / name).read_bytes() == (tmp_path / "run" / "checkpoint-2" / name).read_bytes()


def test_checkpoint_without_processor_files_needs_an_explicit_source(stub_vdpm, tmp_path):
    from dynthinker.train.export_ema import export_ema_checkpoint

    base = _tiny_base(tmp_path / "base")
    _, processor, _ = _run(base, tmp_path / "run", stub_vdpm, geo_aware=False, max_steps=1)
    # A checkpoint as written before processor files were saved with every checkpoint.
    old = tmp_path / "old" / "checkpoint-1"
    shutil.copytree(tmp_path / "run" / "checkpoint-1", old)
    for name in PROCESSOR_FILENAMES:
        (old / name).unlink()

    with pytest.raises(FileNotFoundError, match="--processor-from"):
        export_main(["--checkpoint", str(old), "--output", str(tmp_path / "refused")])
    assert not (tmp_path / "refused").exists()
    with pytest.raises(FileNotFoundError, match="has no preprocessor_config.json"):
        export_main(["--checkpoint", str(old), "--output", str(tmp_path / "refused"), "--processor-from", str(old)])
    with pytest.raises(ValueError, match="already contains"):
        export_main(["--checkpoint", str(tmp_path / "run" / "checkpoint-1"), "--output", str(tmp_path / "refused"),
                     "--processor-from", str(base)])
    assert not (tmp_path / "refused").exists()

    recovered = tmp_path / "recovered"
    assert export_main(["--checkpoint", str(old), "--output", str(recovered), "--processor-from", str(base)]) == 0
    from dynthinker.model.processing_qwen3_vl import Qwen3VLProcessor

    loaded = Qwen3VLProcessor.from_pretrained(str(recovered))
    assert loaded.chat_template == PROCESSOR_TEMPLATE  # not the tokenizer template left in the checkpoint
    assert loaded.video_processor.to_dict() == processor.video_processor.to_dict()
    assert loaded.image_processor.size == {"shortest_edge": 65536, "longest_edge": 16777216}  # base limits

    # The tensor-level API keeps working without processor files unless they are required.
    summary = export_ema_checkpoint(str(old), str(tmp_path / "weights-only"))
    assert summary["replaced_tensors"] > 0
    assert not (tmp_path / "weights-only" / "preprocessor_config.json").exists()


def test_processor_with_named_templates_is_rejected(stub_vdpm, tmp_path):
    from types import SimpleNamespace

    from dynthinker.train.trainer import DynThinkerTrainer

    with pytest.raises(ValueError, match="several named chat templates"):
        DynThinkerTrainer(model=None, processor=SimpleNamespace(chat_template={"default": "a", "tool_use": "b"}))


@pytest.mark.skipif(not os.environ.get("DYNTHINKER_TEST_TOKENIZER"), reason="set DYNTHINKER_TEST_TOKENIZER=<Qwen3-VL dir>")
def test_train_export_and_pipeline_with_real_processor_files(stub_vdpm, tmp_path, monkeypatch):
    """`dynthinker-train` -> checkpoint-<step> -> `dynthinker-export-ema` -> DynThinkerPipeline, no files added."""
    import transformers

    from dynthinker.inference import SYSTEM_PROMPT, DynThinkerPipeline, prepare_frames
    from dynthinker.model.processing_qwen3_vl import Qwen3VLProcessor
    from dynthinker.train import train as train_module

    source = os.environ["DYNTHINKER_TEST_TOKENIZER"]
    base = tmp_path / "base"
    torch.manual_seed(0)
    transformers.Qwen3VLForConditionalGeneration(tiny_config(use_geometry_encoder=False)).save_pretrained(base)
    for name in TOKENIZER_FILES + ("chat_template.json", "preprocessor_config.json", "video_preprocessor_config.json",
                                   "generation_config.json"):
        if os.path.isfile(os.path.join(source, name)):
            shutil.copy2(os.path.join(source, name), base / name)
    examples = tmp_path / "examples"
    shutil.copytree(os.path.join(HERE, os.pardir, "examples"), examples)
    subprocess.run([sys.executable, str(examples / "make_synthetic_media.py")], check=True)

    max_pixels, min_pixels = 576 * 28 * 28, 16 * 28 * 28
    run = tmp_path / "run"

    def train(max_steps):
        monkeypatch.setattr(sys, "argv", [
            "dynthinker-train", "--model_name_or_path", str(base), "--attn_implementation", "eager",
            "--tune_mm_llm", "True", "--dataset_config", str(examples / "datasets.example.json"),
            "--dataset_use", "toy_video,toy_multiview", "--output_dir", str(run), "--use_cpu", "True",
            "--per_device_train_batch_size", "1", "--learning_rate", "1e-3", "--lr_scheduler_type", "constant",
            "--model_max_length", "8192", "--max_pixels", str(max_pixels), "--min_pixels", str(min_pixels),
            "--max_steps", str(max_steps), "--save_strategy", "steps", "--save_steps", "2",
            "--dataloader_num_workers", "0", "--report_to", "none", "--seed", "0",
            "--use_geometry_encoder", "True", "--geometry_encoder_path", stub_vdpm, "--geo_cross_attn", "True",
            "--exclude_geometry_encoder_on_save", "True",
            "--ema_enable", "True", "--ema_start_step", "1", "--ema_update_interval", "1",
            "--ema_mlp_decay", "0.9", "--ema_other_decay", "0.8", "--ema_checkpoint_save", "True",
        ])
        train_module.train()

    train(4)
    base_processor = Qwen3VLProcessor.from_pretrained(str(base))
    for step in (2, 4):
        assert set(PROCESSOR_FILENAMES) <= {p.name for p in (run / f"checkpoint-{step}").iterdir()}

    exported = tmp_path / "exported-step2"
    assert export_main(["--checkpoint", str(run / "checkpoint-2"), "--output", str(exported)]) == 0
    pipeline = DynThinkerPipeline(str(exported), geometry_encoder_path=stub_vdpm, device="cpu", dtype="float32",
                                  attn_implementation="eager")
    assert pipeline.processor.chat_template == base_processor.chat_template
    saved = json.loads((exported / "preprocessor_config.json").read_text())
    assert (saved["max_pixels"], saved["min_pixels"]) == (max_pixels, min_pixels)
    assert saved["size"] == {"longest_edge": max_pixels, "shortest_edge": min_pixels}
    base_video = json.loads((base / "video_preprocessor_config.json").read_text())
    saved_video = json.loads((exported / "video_preprocessor_config.json").read_text())
    assert {k: saved_video[k] for k in base_video} == base_video
    assert pipeline.processor.video_processor.to_dict() == base_processor.video_processor.to_dict()

    # The export builds exactly the inputs that the base model's processor builds.
    frames = synthetic_frames(3)
    inputs = pipeline.build_inputs(frames, "What moves?")
    reference = Qwen3VLProcessor.from_pretrained(str(base), max_pixels=max_pixels, min_pixels=min_pixels,
                                                 padding_side="left")
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [{"type": "image", "image": f} for f in frames]
                 + [{"type": "text", "text": "What moves?"}]}]
    text = reference.apply_chat_template([messages], tokenize=False, add_generation_prompt=True)
    images, _ = prepare_frames(frames, reference.image_processor.patch_size, reference.image_processor.merge_size)
    expected = reference(text=text, images=images, videos=None, padding=True, return_tensors="pt", do_rescale=False)
    for key in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw"):
        assert torch.equal(inputs[key], expected[key]), key
    assert isinstance(pipeline.generate(frames, "What moves?", max_new_tokens=3), str)

    for name in PROCESSOR_FILENAMES:
        assert (run / name).read_bytes() == (run / "checkpoint-4" / name).read_bytes(), name

    train(6)  # auto-resumes from checkpoint-4 with its online weights and D-EMA state
    assert set(PROCESSOR_FILENAMES) <= {p.name for p in (run / "checkpoint-6").iterdir()}
    assert json.loads((run / "checkpoint-6" / "ema_state.json").read_text())["num_updates"] == 6
