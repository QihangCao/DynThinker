# DynThinker Technical Reference

[Back to the project overview](../README.md)

## Scope

| | Status |
|---|---|
| Backbone for DynThinker (VDPM + SDF) | **Qwen3-VL**, using Qwen3-VL-8B-Instruct as the base model. |
| Qwen2.5-VL | D-EMA-only fine-tuning with the stock `transformers` class (`USE_SDF=0`); no SDF. |
| Geometry encoder | V-DPM dual stream (encoder + "diagonal" decoder, layer offset -2). Every frame is decoded; there is no sparse-decoding option. |
| SDF variant | `geo_inject_version="v1"` only. Parameter names match the research checkpoints (`...layers.<i>.cross_attn.*`). |
| DynThinker weights | Not released with this repository. |
| Training records | Images, frame directories and videos with conversations ([docs/DATA.md](DATA.md)). SPAR-7M `spar_info` markers are **not** drawn; datasets that use them are rejected. |

## Configuration notes

- **VDPM decoding.** Every input frame is decoded. Subset decoding with temporal interpolation is
  not provided by this wrapper.
- **Frames and D-EMA decays.** The training launcher defaults to 8 frames and derives the decays from
  a drift budget. Override settings through `EXTRA_ARGS`, for example
  `EXTRA_ARGS="--video_max_frames 32 --ema_mlp_decay 0.999834 --ema_other_decay 0.999834"`.
- **D-EMA boundary.** The late-MLP group uses 0-based layer indices
  `>= --ema_mlp_layer_threshold` (default 4). See [D-EMA](#d-ema).
- **SPAR markers.** Render point/box markers into the images before training and remove `spar_info`
  from the annotations. See [docs/DATA.md](DATA.md).

## Installation

Python >= 3.10, Linux. Install a PyTorch build (>= 2.6) for your CUDA version first, then:

```bash
pip install -e .                 # inference / evaluation
pip install -e ".[train]"        # + DeepSpeed, TensorBoard for training
pip install -e ".[flash]"        # optional flash-attn (Qwen backbone only)
pip install -e ".[eval]"         # pandas/pyarrow for .parquet benchmark annotations
pip install -e ".[dev]"          # pytest
```

`transformers` is pinned to 4.57.0 because `model/modeling_qwen3_vl.py` is a modified copy of that
version's Qwen3-VL code.

## Weights you need

| Weight | Where | Used for |
|---|---|---|
| Qwen3-VL-Instruct, e.g. `Qwen/Qwen3-VL-8B-Instruct` | Hugging Face Hub | training (initialisation) |
| V-DPM checkpoint (`model.pt`, plain state dict) | the V-DPM project, https://github.com/eldar/vdpm | training; inference if the DynThinker checkpoint was saved without VDPM weights |
| A trained DynThinker checkpoint | produced by `dynthinker-train` | inference / evaluation |

Check the license terms of each checkpoint (see [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)).

## Inference

```bash
dynthinker-infer --model /path/to/dynthinker-checkpoint \
    --video clip.mp4 --num-frames 32 \
    --question "Is the car moving towards the camera? Answer yes or no."
# or an ordered list of frames:
dynthinker-infer --model CKPT --images f0.jpg f1.jpg f2.jpg f3.jpg --question "..."
```

```python
from dynthinker.inference import DynThinkerPipeline, load_frames

pipe = DynThinkerPipeline("/path/to/dynthinker-checkpoint", device="cuda")
frames = load_frames(video="clip.mp4", num_frames=32)
print(pipe.generate(frames, "How many chairs are in the room? Answer with a number."))
```

- Frames are sampled uniformly (default 32, pixel limits 12544-451584 per frame) and are fed as a
  sequence of images; each frame also goes to VDPM.
- Greedy decoding by default (`--temperature 0`).
- The pipeline refuses checkpoints that are not DynThinker checkpoints (for example a plain Qwen3-VL
  checkpoint, or a config without `use_geometry_encoder`, `geo_cross_attn`, `geo_inject_version="v1"`
  and the dual-stream VDPM flags), checkpoints missing SDF/geometry-merger weights, and checkpoints
  saved without VDPM weights unless `--geometry-encoder-path` (or `DYNTHINKER_VDPM_PATH`) is given.
- A GPU is expected. The 8B model plus VDPM (~1.2B parameters) does not fit a typical CPU setup
  comfortably in float32.

## Training

1. Prepare data and a dataset registry: see [docs/DATA.md](DATA.md).
2. Launch (one node, 8 GPUs, DeepSpeed ZeRO-2, bf16):

```bash
export MODEL_PATH=Qwen/Qwen3-VL-8B-Instruct
export VDPM_PATH=/path/to/vdpm/model.pt
export DATASET_CONFIG=/path/to/datasets.yaml
export DATASETS="my_spatial_set,my_video_set%50"
export DATASET_SIZE=200000          # samples after %-sampling; used to derive the D-EMA decays
export OUTPUT_DIR=./output/dynthinker
DRY_RUN=1 bash scripts/train_dynthinker.sh   # print the torchrun command
bash scripts/train_dynthinker.sh
```

The script documents all environment variables in its header. It trains the LLM (with SDF) and
keeps the vision tower, the vision merger, VDPM and the geometry merger frozen, uses a global batch of
64, enables token compression for samples with more than 8 frames (`TOKEN_COMPRESSION=1`, recorded in
the checkpoint and applied again at inference), and enables D-EMA with checkpointed EMA state.
Any extra `dynthinker-train` argument can be appended with `EXTRA_ARGS="..."`
(later arguments override earlier ones).

`dynthinker-train --help` lists every option. Notes:

- The geometry merger (VDPM 2048-d features to the LLM width) is initialised randomly and frozen by
  the default launcher; SDF learns its own projections on top. The merger is saved with the
  checkpoint. Pass `--tune_geometry_merger True` to train it.
- SDF modules are inserted in layers `num_layers/4 < i < num_layers - num_layers/4` (0-based; layers
  10-26 for the 36-layer 8B model).
- `--exclude_geometry_encoder_on_save True` omits the frozen VDPM weights from checkpoints. Such
  checkpoints need `--geometry-encoder-path` at inference time. In Python, load VDPM weights with
  `model.load_vdpm_weights(path)` after `from_pretrained`.

## D-EMA

Implemented in [`src/dynthinker/train/dema.py`](../src/dynthinker/train/dema.py) as a `TrainerCallback`.

**Groups** (trainable parameters only, by name):

| Group | Rule | Decay option | Applied at the end |
|---|---|---|---|
| `mlp` (sensitive) | `*.language_model.layers.<i>.mlp.*` with **0-based** `i >= --ema_mlp_layer_threshold` | `--ema_mlp_decay` | yes |
| `new_module` | name contains `cross_attn` (SDF) or `geometry_merger` | `--ema_new_module_decay` (only tracked) | no (online weights kept) |
| `other` (base) | everything else (attention, early MLPs, norms, embeddings, `lm_head`, trainable visual parts) | `--ema_other_decay` | yes |

With the default threshold 4, 0-based layers 4, 5, ... form the sensitive group (layers 5 onward
with 1-based numbering). Set `--ema_mlp_layer_threshold` to change the boundary. Training checks
that late-layer MLP parameters match the grouping rule.

**Schedule.** Shadows start from the parameters at the beginning of training (the pretrained
weights). After every optimizer step `s >= 1` with `s >= --ema_start_step` and
`(s - start) % --ema_update_interval == 0`, each shadow is updated as
`shadow = rho * shadow + (1 - rho) * online`. `--ema_start_step 0` therefore first updates at step
`--ema_update_interval`.

**Decay from a drift budget.** `rho = (1 - delta)^(1/N)` for `N` updates keeps a fraction `delta`
of the final shadow on post-training states. The scale-normalised horizon keeps
`rho^N` constant across schedules. `dynthinker-ema-decay` computes both, and
`scripts/train_dynthinker.sh` uses it to derive the decays from `DEMA_DRIFT_MLP` /
`DEMA_DRIFT_OTHER` (default 0.125 each) and `DATASET_SIZE`. The update count `N` is estimated as
`ceil(DATASET_SIZE / 64)` steps. The Trainer's actual step count can be slightly lower (partial batches
and incomplete modality-grouped mega-batches are dropped), so the callback logs the realised `N` and
per-group drift at the start of training.

The launcher uses start step 150, interval 10 and threshold 4; `new_module` keeps its online weights
at the end. To use fixed decays instead of the drift budget, pass, for example,
`EXTRA_ARGS="--ema_mlp_decay 0.9998 --ema_other_decay 0.9998"`.

**Checkpoints, resume and export.**

- Trainer checkpoints (`OUTPUT_DIR/checkpoint-<step>/`) contain the **online** weights plus
  `ema_state.safetensors` / `ema_state.json` (shadows, update count, D-EMA configuration) when
  `--ema_checkpoint_save True` (set by the launcher; the CLI default is `False`).
- Resuming (automatic from the latest `checkpoint-<step>`, or `--resume_from_checkpoint`) restores
  the online weights, optimizer state and shadows. With `--ema_checkpoint_strict True` (default) a
  missing shadow, a different D-EMA configuration, different parameter names/shapes or an
  inconsistent update count is an error. Resuming a D-EMA run whose checkpoints hold no EMA state
  (`--ema_checkpoint_save False`) is also an error, because the shadows would restart from the resumed
  weights instead of the pretrained ones. `--ema_reinit_on_resume True` accepts that explicitly.
- At the end of training the `mlp` and `other` shadows are copied into the model and the result is
  saved to `OUTPUT_DIR` with a `dema_export.json` marker (`resumable: false`). This final directory is
  for inference, and `dynthinker-train` refuses to resume from it.
- Each checkpoint and the final `OUTPUT_DIR` also save the image processor configuration with the
  training pixel limits, the video processor configuration, and the effective processor chat template.
- `dynthinker-export-ema --checkpoint OUTPUT_DIR/checkpoint-2000 --output EXPORTS/dema-step2000`
  applies the same consolidation to an intermediate checkpoint. The export contains config,
  tokenizer/processor files and averaged weights and loads directly with `DynThinkerPipeline`.
  It has no optimizer, scheduler, RNG, trainer or EMA state and carries a `dema_export.json` marker;
  `dynthinker-train` refuses to resume from it.
- Older checkpoints missing processor files require an explicit `--processor-from DIR`. Use the final
  `OUTPUT_DIR` of the same run, or the base model directory used for training. The same-run directory
  preserves training pixel limits; the base directory restores its original limits. Older final
  directories may lack video configuration, in which case Transformers derives it from the image
  configuration. No processor source is selected automatically.

**Precision and distributed training.** Shadows are kept in float32 (CPU by default, `--ema_on_gpu True` for GPU) and track
the module parameters as seen by the Trainer; under bf16 DeepSpeed training these are the bf16 module
weights. `--ema_rank_zero_only True` (used by the script) keeps shadows on
global rank 0 only, which is valid for data parallelism and ZeRO-1/2 where every rank holds full
parameters. DeepSpeed ZeRO-3 (partitioned parameters) is not supported. Tied input/output embeddings
appear once in `named_parameters()` and are averaged once, in the `other` group.

## Evaluation

`dynthinker-eval` runs VSI-Bench or VLM4D from local files with the same prompts and scoring as the
research code's lmms-eval tasks, without the lmms-eval framework:

```bash
dynthinker-eval generate --benchmark vsibench --annotations VSI-Bench/test.jsonl \
    --media-root VSI-Bench --model CKPT --output preds_vsibench.jsonl
dynthinker-eval score --benchmark vsibench --predictions preds_vsibench.jsonl
```

- VSI-Bench videos are expected at `<media-root>/<dataset>/<scene_name>.mp4`; VLM4D videos at
  `<media-root>/<path in the "video" field>` (a Hub URL prefix up to `resolve/main/` is stripped).
- Generation is single-process and appends to the predictions file, so an interrupted run resumes.
  Each record carries a fingerprint of the run (annotation hash, media root, checkpoint config and weight
  digests, VDPM file digest, frame/pixel/decoding settings), which is also stored in `<output>.run.json`.
  A restart with any different setting, or with annotations that no longer match the stored records in
  order, is refused. `score` rejects duplicate indices and files that mix runs.
- Benchmark data must be downloaded from the benchmark providers.

Evaluation requires a local checkpoint directory containing model weights; download Hub checkpoints first.

Checkpoint identity uses full SHA-256 hashes of the model weights, external VDPM weights and
recognized model, processor and tokenizer assets. Startup reads each weight file once, so hashing
large checkpoints can take time. Unrelated files in the checkpoint directory are ignored.

VDPM uses the upstream VGGT/DINOv2 PyTorch SDPA attention implementation. The optional
`flash-attn` extra applies to the Qwen backbone. Different attention backends and dtypes can produce
small numerical differences.

## Tests

```bash
pip install -e ".[dev]"
pytest -q                                   # CPU, tiny random models, no downloads
DYNTHINKER_TEST_TOKENIZER=/path/to/Qwen3-VL-8B-Instruct pytest -q   # + tokenizer-based data/inference tests
DYNTHINKER_SLOW_TESTS=1 pytest -q tests/test_vdpm_encoder.py         # real VDPM architecture, ~26 GB RAM
```

The tokenizer-based tests read only tokenizer/processor files from that directory, not model weights.

The CPU test suite uses tiny random models to cover SDF forward/backward, generation, checkpoint
round trips, D-EMA grouping and updates, resume/export, and data preprocessing. Optional tests cover
the tokenizer/processor pipeline and the full V-DPM architecture with random weights.

## License

Mixed. Code written for DynThinker is MIT ([LICENSE](../LICENSE)). Code derived from other projects
keeps its original license (Apache-2.0, MIT, or the VGGT License for the V-DPM/VGGT-derived parts).
See [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) for the per-file map and [LICENSES/](../LICENSES)
for the texts.

## Acknowledgements

DynThinker builds on [GeoThinker](https://github.com/Li-Hao-yuan/GeoThinker),
[VG-LLM](https://github.com/LaVi-Lab/VG-LLM), [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL),
[V-DPM](https://github.com/eldar/vdpm), [VGGT](https://github.com/facebookresearch/vggt),
[Transformers](https://github.com/huggingface/transformers),
[lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) and
[thinking-in-space](https://github.com/vision-x-nyu/thinking-in-space).
