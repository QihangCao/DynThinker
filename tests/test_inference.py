"""Inference pipeline: checkpoint guards, frame preparation and (optionally) end-to-end generation."""

import os
import shutil

import pytest
import torch
from safetensors.torch import load_file, save_file

from conftest import synthetic_frames, tiny_config
from dynthinker.inference import DynThinkerPipeline, check_loaded_weights, prepare_frames


def test_plain_qwen3_vl_checkpoint_is_rejected(tmp_path):
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig

    for config in (Qwen3VLConfig(), tiny_config(use_geometry_encoder=False)):
        config.save_pretrained(tmp_path)
        with pytest.raises(ValueError, match="not a DynThinker"):
            DynThinkerPipeline(str(tmp_path), device="cpu", dtype="float32")


def test_wrong_sdf_version_is_rejected(tmp_path):
    config = tiny_config()
    config.geo_inject_version = "v31"
    config.save_pretrained(tmp_path)
    with pytest.raises(ValueError, match="geo_inject_version='v31'"):
        DynThinkerPipeline(str(tmp_path), device="cpu", dtype="float32")


def test_missing_fusion_or_vdpm_weights_are_rejected(stub_geometry, tmp_path):
    torch.manual_seed(0)
    model = stub_geometry.Qwen3VLForConditionalGenerationWithVGGT(tiny_config())
    model.save_pretrained(tmp_path)
    _, info = stub_geometry.Qwen3VLForConditionalGenerationWithVGGT.from_pretrained(
        str(tmp_path), output_loading_info=True
    )
    check_loaded_weights(info["missing_keys"], geometry_encoder_loaded=False)

    weights = load_file(str(tmp_path / "model.safetensors"))
    save_file({k: v for k, v in weights.items() if "cross_attn" not in k}, str(tmp_path / "model.safetensors"),
              metadata={"format": "pt"})
    _, info = stub_geometry.Qwen3VLForConditionalGenerationWithVGGT.from_pretrained(
        str(tmp_path), output_loading_info=True
    )
    with pytest.raises(ValueError, match="SDF/geometry-merger"):
        check_loaded_weights(info["missing_keys"], geometry_encoder_loaded=False)

    vdpm_missing = ["model.geometry_encoder.vdpm.aggregator.camera_token"]
    with pytest.raises(ValueError, match="geometry-encoder-path"):
        check_loaded_weights(vdpm_missing, geometry_encoder_loaded=False)
    check_loaded_weights(vdpm_missing, geometry_encoder_loaded=True)


def test_prepare_frames_aligns_qwen_and_vdpm_inputs():
    frames = synthetic_frames(3, size=(640, 360)) + synthetic_frames(1, size=(480, 480))
    qwen_images, geometry = prepare_frames(frames, patch_size=16, merge_size=2)
    assert geometry.shape[0] == 4 and geometry.shape[1] == 3 and geometry.shape[-1] == 518
    assert geometry.shape[-2] % 14 == 0 and geometry.shape[-1] % 14 == 0
    for image in qwen_images:
        # Cropped to a whole number of merge groups of patches; the Qwen processor resizes afterwards.
        assert (image.shape[-2] // 16) % 2 == 0 and (image.shape[-1] // 16) % 2 == 0
        assert image.shape[-2] <= geometry.shape[-2] and image.shape[-1] <= geometry.shape[-1]
    assert float(geometry.min()) >= 0.0 and float(geometry.max()) <= 1.0


@pytest.mark.skipif(not os.environ.get("DYNTHINKER_TEST_TOKENIZER"), reason="set DYNTHINKER_TEST_TOKENIZER=<Qwen3-VL dir>")
def test_pipeline_generates_with_tiny_model(stub_geometry, tmp_path):
    """Tiny random weights + the real Qwen3-VL tokenizer/processor files (no model weights are read)."""
    source = os.environ["DYNTHINKER_TEST_TOKENIZER"]
    torch.manual_seed(0)
    stub_geometry.Qwen3VLForConditionalGenerationWithVGGT(tiny_config()).save_pretrained(tmp_path)
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.json",
                 "preprocessor_config.json", "video_preprocessor_config.json", "generation_config.json"):
        if os.path.isfile(os.path.join(source, name)):
            shutil.copy2(os.path.join(source, name), tmp_path / name)

    pipeline = DynThinkerPipeline(str(tmp_path), device="cpu", dtype="float32", attn_implementation="eager")
    frames = synthetic_frames(3)
    inputs = pipeline.build_inputs(frames, "What moves?")
    assert inputs["geometry_encoder_inputs"].shape[:2] == (1, 3)
    assert inputs["image_grid_thw"].shape == (3, 3)
    num_image_tokens = int((inputs["input_ids"] == pipeline.model.config.image_token_id).sum())
    assert num_image_tokens == int(inputs["image_grid_thw"].prod(dim=-1).sum()) // 4
    answer = pipeline.generate(frames, "What moves?", max_new_tokens=3)
    assert isinstance(answer, str)
    assert pipeline.generate(frames, "What moves?", max_new_tokens=3) == answer  # greedy is deterministic


@pytest.mark.parametrize("patch,merge,shape", [(14, 2, (280, 504)), (16, 2, (294, 518)), (16, 3, (294, 486))])
def test_evaluation_crop_keeps_subpatch_pixels(patch, merge, shape):
    frames = synthetic_frames(1, size=(640, 360)) + synthetic_frames(1, size=(480, 480))
    images, geometry = prepare_frames(frames, patch, merge)
    assert tuple(geometry.shape) == (2, 3, 294, 518)
    rows, columns = shape
    for position, image in enumerate(images):
        assert tuple(image.shape[-2:]) == shape
        assert torch.equal(image, geometry[position, :, :rows, :columns])
    # The geometry batch is independent of the Qwen views.
    snapshot = geometry.clone()
    images[0].zero_()
    assert torch.equal(geometry, snapshot)
