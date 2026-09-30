# Third-party notices

This repository mixes code written for DynThinker with code derived from other projects.
Files written for DynThinker are under the MIT License in [`LICENSE`](LICENSE). Every other file keeps
the license of the project it comes from. The table below lists each derived file or directory, where
it comes from, and which license applies. Full license texts are in [`LICENSES/`](LICENSES) and, for
vendored directories, next to the code. The per-file headers are authoritative.

The repository as a whole is **not** MIT-licensed. In particular, the V-DPM/VGGT-derived code under
`src/dynthinker/model/vdpm/`, `src/dynthinker/model/vggt/heads/` and
`src/dynthinker/data/image_utils.py` is distributed under the VGGT License, which includes an
Acceptable Use Policy and requires that you pass a copy of that license on with any redistribution.

## License map

| Path (under `src/dynthinker/` unless noted) | Origin | License |
|---|---|---|
| `model/modeling_qwen3_vl.py`, `model/processing_qwen3_vl.py` | HuggingFace Transformers 4.57.0, Qwen3-VL model/processor (Copyright 2025 The Qwen Team and The HuggingFace Inc. team), modified by the GeoThinker authors (MIT) and the DynThinker authors | Apache-2.0 ([`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt)); GeoThinker modifications also MIT ([`LICENSES/MIT-GeoThinker.txt`](LICENSES/MIT-GeoThinker.txt)) |
| `model/vdpm/model.py` | V-DPM, https://github.com/eldar/vdpm (Copyright (c) 2025 Eldar Insafutdinov, Edgar Sucar), modified | MIT ([`model/vdpm/LICENSE`](src/dynthinker/model/vdpm/LICENSE), [`LICENSES/MIT-VDPM.txt`](LICENSES/MIT-VDPM.txt)) |
| `model/vdpm/aggregator.py`, `model/vdpm/decoder.py` | V-DPM; Copyright (c) Meta Platforms, Inc. and affiliates (VGGT-derived); import paths changed | VGGT License ([`model/vdpm/LICENSE-VGGT`](src/dynthinker/model/vdpm/LICENSE-VGGT)) |
| `model/vggt/heads/` | VGGT, https://github.com/facebookresearch/vggt (Copyright (c) Meta Platforms, Inc. and affiliates); import paths changed, one comment edited in `camera_head.py` | VGGT License ([`model/vggt/LICENSE`](src/dynthinker/model/vggt/LICENSE), [`LICENSES/VGGT-LICENSE.txt`](LICENSES/VGGT-LICENSE.txt)) |
| `model/vggt/layers/` | VGGT, derived from DINOv2 (Copyright (c) Meta Platforms, Inc. and affiliates). `attention.py` is an unchanged copy from VGGT commit `a288dd0f14786c93483e45524328726ab7b1b4ce`; it uses PyTorch SDPA | Apache-2.0 ([`model/vggt/layers/LICENSE`](src/dynthinker/model/vggt/layers/LICENSE)) |
| `data/image_utils.py` | Crop mode of `load_and_preprocess_images` from VGGT `vggt/utils/load_fn.py`, reduced to one frame (PIL image or path) by the DynThinker authors | VGGT License |
| `data/registry.py` | DynThinker dataset registry; `parse_sampling_rate` adapted from [Qwen-VL fine-tuning](https://github.com/QwenLM/Qwen3-VL/blob/271ad06/qwen-vl-finetune/qwenvl/data/__init__.py) | MIT (DynThinker); sampling-rate helper Apache-2.0 |
| `data/rope2d.py` | Qwen-VL fine-tuning code, https://github.com/QwenLM/Qwen3-VL (Alibaba Cloud), commit 271ad06; an unused import removed | Apache-2.0 |
| `data/dataset.py` | Qwen-VL fine-tuning code (Apache-2.0): tokenisation/loss masking, sampling rates, retries, collation core. Modified by the DynThinker authors (see file header) | Apache-2.0 |
| `train/train.py`, `train/arguments.py` | Qwen-VL fine-tuning code (Apache-2.0; adopts code from FastChat and Stanford Alpaca, Apache-2.0); `group_by_modality_length` from LLaVA (Apache-2.0); modified by the DynThinker authors | Apache-2.0 |
| `train/trainer.py` | Qwen-VL fine-tuning code and the length-grouped samplers of LLaVA, https://github.com/haotian-liu/LLaVA, and LLaVA-NeXT, https://github.com/LLaVA-VL/LLaVA-NeXT (all Apache-2.0), modified | Apache-2.0 |
| `video.py` | lmms-eval, https://github.com/EvolvingLMMs-Lab/lmms-eval, `load_video.py`, reduced | Apache-2.0 |
| `eval/benchmarks.py` | VSI-Bench prompting/scoring from thinking-in-space, https://github.com/vision-x-nyu/thinking-in-space (Apache-2.0); VLM4D task from the GeoThinker/DynThinker lmms-eval tasks; reimplemented | Apache-2.0 |
| `inference.py` | DynThinker pipeline; resizing helper adapted from GeoThinker (MIT). Evaluation crop and processor input assembly implement the tensor conventions of VG-LLM/GeoThinker using the Transformers processor API | MIT (DynThinker and GeoThinker) |
| `model/geometry_merger.py` | Written for this release (imports `Qwen2RMSNorm` from HuggingFace Transformers, Apache-2.0) | MIT ([`LICENSE`](LICENSE)) |
| `configs/deepspeed_zero2.json` (repository root) | GeoThinker `scripts/zero2_opt.json`, unmodified | MIT (GeoThinker) |
| Everything else (`model/sdf.py`, `model/geometry_encoders/`, `data/preprocess.py`, `train/dema.py`, `train/export_ema.py`, `tools/`, `infer.py`, `eval/run.py`, `tests/`, `scripts/`, `configs/datasets.example.yaml`, `examples/`) | Written for DynThinker | MIT ([`LICENSE`](LICENSE)) |

GeoThinker: https://github.com/Li-Hao-yuan/GeoThinker, Copyright (c) 2026 sysu19351063, MIT License.

## Model weights

No model weights are included. Weights you download separately have their own terms:

- Qwen3-VL checkpoints: see the license on the respective Qwen model card.
- V-DPM checkpoint (`model.pt`): see the V-DPM project. V-DPM builds on VGGT, whose code and weights
  are covered by the VGGT License.

The VDPM attention implementation comes from
[VGGT at the pinned revision](https://github.com/facebookresearch/vggt/blob/a288dd0f14786c93483e45524328726ab7b1b4ce/vggt/layers/attention.py), with its original Apache-2.0 notice.
