# Training data format

No training data is distributed with this repository. You provide annotation files and media, and
list them in a dataset registry.

## Dataset registry (`--dataset_config`)

A YAML or JSON mapping from dataset name to entry:

```yaml
my_spatial_set:
  annotation_path: ${DATA_ROOT}/train/my_spatial_set.jsonl  # .json (list of records) or .jsonl
  data_path: ${DATA_ROOT}/media                            # root for relative media paths in the records
  tag: 3d                                                  # 3d or 2d (default 2d)
```

- Relative paths are resolved against the registry file's directory. `$VARS` / `${VARS}` are expanded;
  an unset variable is an error.
- `tag` only matters with `--group_by_modality_length True`: a batch never mixes `3d` and `2d`
  samples, and the last incomplete mega-batch of each tag is dropped every epoch (behaviour of the
  original sampler).
- Select datasets with `--dataset_use name1,name2%50` (`%N` keeps a random N% of that dataset).

[`configs/datasets.example.yaml`](../configs/datasets.example.yaml) lists the dataset names used by the
research training scripts as a template. The annotation files themselves must be obtained from the
respective dataset providers.

## Records

Each record has `conversations` and, for visual samples, `images` (or `image`) or `video`. If a record
has several of these, `images` wins over `video`, which wins over `image`. Records without any are text-only:

```json
{
  "id": "sample-0",
  "video": "clips/clip_000",
  "conversations": [
    {"from": "human", "value": "<video>\nIs the red square moving to the left or to the right?\nOptions:\nA. left\nB. right\nAnswer with the option's letter from the given choices directly."},
    {"from": "gpt", "value": "B"}
  ]
}
```

```json
{
  "id": "sample-1",
  "images": ["scene/view_0.jpg", "scene/view_1.jpg", "scene/view_2.jpg"],
  "conversations": [
    {"from": "human", "value": "<image>\n<image>\n<image>\nWhich object is closer to the camera in the first view, the chair or the table?"},
    {"from": "gpt", "value": "The chair."}
  ]
}
```

- `conversations`: alternating turns with `from` in `human`/`gpt` (or `user`/`assistant`; an initial
  `system` turn replaces the default system prompt). `value` may also be given as `content`. Only
  assistant turns are supervised; system/user tokens and the `<|im_start|>assistant\n` prefix are
  masked with `-100`.
- `images`: paths relative to `data_path`. The human turn must contain one `<image>` per image. All
  images are resized to the size of the first one.
- `video`: a directory of frame images (sorted by file name, treated as 1 frame per second) or an
  `.mp4`/`.avi`/`.mov` file. Exactly one turn must contain `<video>`, which is replaced by one `<image>`
  per sampled frame. Frames are sampled uniformly, one every `--base_interval` seconds, clamped to
  `[--video_min_frames, --video_max_frames]`. Video files are decoded with Decord if it is installed,
  otherwise with PyAV.
- For SPAR-7M-style data such as `spar_234k`, render point/box markers into the image files during
  preprocessing, then remove the `spar_info` field. Use a rendering tool whose license permits your
  intended use. The loader reads the prepared images and rejects records that still contain
  `spar_info`; it does not draw markers at training time.

Every frame is preprocessed once for both branches: it is resized to width 518 (height a multiple of
14, center-cropped to at most 518) for VDPM, and the same tensor, cropped to a whole number of
`patch_size * merge_size` blocks, goes through the Qwen3-VL image processor. Frames are always fed as
images (not as a Qwen video), so VDPM frame `t` and Qwen image `t` are aligned.

With `--depart_smi_token True` (token compression), samples with more than `--smi_image_num` frames use
a merge size of `2 * smi_downsample_rate`, i.e. 4x fewer visual tokens per frame.

## Toy example

```bash
python examples/make_synthetic_media.py           # writes examples/toy_media/ (synthetic images)
# registry: examples/datasets.example.json, records: examples/data/*.json
```

The toy records exist to check that the pipeline runs; they are not meaningful training data.
