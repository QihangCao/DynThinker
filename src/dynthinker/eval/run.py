# Copyright (c) 2026 The DynThinker Authors. Licensed under the MIT License.
"""`dynthinker-eval`: single-process evaluation on VSI-Bench or VLM4D from local files.

Generation mode (needs a GPU for real checkpoints):
    dynthinker-eval generate --benchmark vsibench --annotations VSI-Bench/test.jsonl \
        --media-root VSI-Bench --model CKPT --output preds.jsonl
Scoring mode (CPU only; re-scores a predictions file):
    dynthinker-eval score --benchmark vsibench --predictions preds.jsonl

Annotations may be .jsonl, .json (list) or .parquet (requires pandas + pyarrow).
An interrupted run can be restarted with the same --output. Every record carries a fingerprint of the
run settings (also written to ``<output>.run.json``); a restart with a different model, VDPM file,
annotations, media root, frame/pixel/decoding settings or benchmark is refused, as is a file whose
records do not match the annotations in order.
"""

import argparse
import hashlib
import json
import os
import sys
from typing import Iterable, List, Optional

from .. import __version__
from .benchmarks import BENCHMARKS



def load_annotations(path: str) -> List[dict]:
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise ValueError(f"{path} must contain a list of records")
        return data
    if path.endswith(".parquet"):
        import pandas as pd

        return pd.read_parquet(path).to_dict(orient="records")
    raise ValueError(f"Unsupported annotation format: {path}")


def _read_predictions(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON; repair the damaged line or use a new --output"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a prediction object")
            records.append(record)
    return records


_MODEL_METADATA_FILES = (
    "config.json", "generation_config.json", "preprocessor_config.json", "processor_config.json",
    "video_preprocessor_config.json", "tokenizer.json", "tokenizer_config.json",
    "special_tokens_map.json", "added_tokens.json", "vocab.json", "vocab.txt", "merges.txt",
    "tokenizer.model", "spiece.model", "sentencepiece.bpe.model", "chat_template.jinja", "chat_template.json",
)
_WEIGHT_INDEX_FILES = ("model.safetensors.index.json", "pytorch_model.bin.index.json")


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _weight_fingerprint(path: str) -> dict:
    """Read the entire weight file: frozen head/tail tensors cannot identify trained middle layers."""
    return {"size": os.path.getsize(path), "sha256": _sha256_file(path)}


def _model_fingerprint(model_dir: str) -> dict:
    """Hash recognized model assets and complete weights; ignore unrelated checkpoint-directory files."""
    if not os.path.isdir(model_dir):
        raise ValueError("--model must be a local checkpoint directory; download Hub checkpoints first")
    metadata = {name for name in _MODEL_METADATA_FILES if os.path.isfile(os.path.join(model_dir, name))}
    weights = {name for name in ("model.safetensors", "pytorch_model.bin")
               if os.path.isfile(os.path.join(model_dir, name))}
    for index_name in _WEIGHT_INDEX_FILES:
        index_path = os.path.join(model_dir, index_name)
        if not os.path.isfile(index_path):
            continue
        metadata.add(index_name)
        with open(index_path, encoding="utf-8") as handle:
            index = json.load(handle)
        mapping = index.get("weight_map")
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError(f"{index_name} must contain a non-empty weight_map")
        for value in mapping.values():
            if not isinstance(value, str):
                raise ValueError(f"{index_name} contains a non-string shard path")
            relative = os.path.normpath(value)
            if os.path.isabs(value) or relative == ".." or relative.startswith(".." + os.sep):
                raise ValueError(f"{index_name} contains a shard path outside the checkpoint directory")
            if not relative.endswith((".safetensors", ".bin")):
                raise ValueError(f"{index_name} contains an unsupported weight-file suffix")
            if not os.path.isfile(os.path.join(model_dir, relative)):
                raise ValueError(f"{index_name} references a missing weight shard")
            weights.add(relative)
    if not weights:
        raise ValueError("The checkpoint directory contains no supported model weights")
    templates = os.path.join(model_dir, "chat_templates")
    if os.path.isdir(templates):
        metadata.update(os.path.join("chat_templates", name) for name in os.listdir(templates)
                        if name.endswith(".jinja") and os.path.isfile(os.path.join(templates, name)))
    files = {name: _sha256_file(os.path.join(model_dir, name)) for name in sorted(metadata)}
    files.update({name: _weight_fingerprint(os.path.join(model_dir, name)) for name in sorted(weights)})
    return files


def run_fingerprint(args, max_new_tokens: int, system_prompt: str) -> dict:
    vdpm_path = args.geometry_encoder_path or os.environ.get("DYNTHINKER_VDPM_PATH") or None
    return {
        "dynthinker_version": __version__,
        "benchmark": args.benchmark,
        "annotations_sha256": _sha256_file(args.annotations),
        "media_root": os.path.realpath(args.media_root),
        "model": _model_fingerprint(args.model),
        "vdpm_weights": _weight_fingerprint(vdpm_path) if vdpm_path else None,
        "num_frames": args.num_frames,
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "max_new_tokens": max_new_tokens,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "system_prompt": system_prompt,
    }


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True)  # NaN-safe comparison of JSON round-tripped docs


def fingerprint_id(fingerprint: dict) -> str:
    return hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def check_resumable(records: List[dict], docs: List[dict], run_id: str, path: str) -> None:
    """Existing records must come from this run and be annotations[0..n-1] in order."""
    for position, record in enumerate(records):
        if record.get("run") != run_id:
            raise ValueError(
                f"{path} was written by a different run (record {position} has run={record.get('run')!r}, "
                f"this run is {run_id!r}; see {path}.run.json). Use a new --output."
            )
        if record.get("index") != position:
            raise ValueError(f"{path}: record {position} has index {record.get('index')!r}; expected {position}")
        if position >= len(docs) or _canonical(record.get("doc")) != _canonical(docs[position]):
            raise ValueError(f"{path}: record {position} does not match annotation {position}; use a new --output")


def check_records(records: List[dict], benchmark: str, sidecar: Optional[dict]) -> None:
    """Reject duplicate indices and predictions mixed from several runs."""
    indices = [record.get("index") for record in records]
    if len(set(indices)) != len(indices):
        raise ValueError("predictions contain duplicate indices")
    runs = {record.get("run") for record in records}
    if len(runs) > 1:
        raise ValueError(f"predictions mix several runs: {sorted(map(str, runs))}")
    if sidecar is not None:
        if sidecar.get("fingerprint", {}).get("benchmark") != benchmark:
            raise ValueError(f"predictions were generated for {sidecar.get('fingerprint', {}).get('benchmark')!r}")
        if runs and runs != {sidecar.get("run")}:
            raise ValueError("predictions do not match their .run.json fingerprint")
    elif runs == {None}:
        print("warning: predictions carry no run fingerprint (not written by dynthinker-eval)", file=sys.stderr)


def _read_sidecar(predictions_path: str) -> Optional[dict]:
    path = predictions_path + ".run.json"
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def score_records(benchmark: str, records: Iterable[dict], sidecar: Optional[dict] = None) -> dict:
    spec = BENCHMARKS[benchmark]
    records = list(records)
    check_records(records, benchmark, sidecar)
    scored = []
    for record in records:
        record = dict(record)
        record["score"] = spec["score"](record["doc"], record["prediction"])
        for field in spec["keep_fields"]:
            if field in record["doc"]:
                record[field] = record["doc"][field]
        scored.append(record)
    if not scored:
        raise ValueError("No predictions to score")
    return {"benchmark": benchmark, "num_samples": len(scored), "metrics": spec["aggregate"](scored)}


def _json_safe(doc: dict) -> dict:
    """Parquet rows may contain numpy values; convert them for JSON output."""
    safe = {}
    for key, value in doc.items():
        if hasattr(value, "tolist"):
            value = value.tolist()
        safe[key] = value
    return safe


def cmd_generate(args) -> int:
    from ..inference import SYSTEM_PROMPT, DynThinkerPipeline, load_frames

    spec = BENCHMARKS[args.benchmark]
    docs = [_json_safe(doc) for doc in load_annotations(args.annotations)]
    if args.limit is not None:
        docs = docs[: args.limit]
    max_new_tokens = args.max_new_tokens or spec["max_new_tokens"]
    fingerprint = run_fingerprint(args, max_new_tokens, SYSTEM_PROMPT)
    run_id = fingerprint_id(fingerprint)

    existing = _read_predictions(args.output)
    sidecar = _read_sidecar(args.output)
    if existing and (sidecar is None or sidecar.get("run") != run_id):
        raise ValueError(f"{args.output} exists but was not written by this run configuration; use a new --output")
    check_resumable(existing, docs, run_id, args.output)
    done = len(existing)
    if done:
        print(f"resuming after {done} existing predictions in {args.output}", file=sys.stderr)
    with open(args.output + ".run.json", "w", encoding="utf-8") as handle:
        json.dump({"run": run_id, "fingerprint": fingerprint}, handle, indent=2, sort_keys=True)

    pipeline = DynThinkerPipeline(
        args.model,
        geometry_encoder_path=args.geometry_encoder_path,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    with open(args.output, "a", encoding="utf-8") as handle:
        for index, doc in enumerate(docs):
            if index < done:
                continue
            video_path = spec["video_path"](doc, args.media_root)
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"video not found: {video_path}")
            frames = load_frames(video=video_path, num_frames=args.num_frames)
            prediction = pipeline.generate(frames, spec["prompt"](doc), max_new_tokens=max_new_tokens)
            handle.write(json.dumps({"index": index, "run": run_id, "doc": doc, "prediction": prediction}) + "\n")
            handle.flush()

    summary = score_records(args.benchmark, _read_predictions(args.output), _read_sidecar(args.output))
    print(json.dumps(summary, indent=2))
    return 0


def cmd_score(args) -> int:
    summary = score_records(args.benchmark, _read_predictions(args.predictions), _read_sidecar(args.predictions))
    text = json.dumps(summary, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    print(text)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dynthinker-eval", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Run a checkpoint and write predictions (JSONL)")
    gen.add_argument("--benchmark", required=True, choices=sorted(BENCHMARKS))
    gen.add_argument("--annotations", required=True, help="Benchmark annotations (.jsonl/.json/.parquet)")
    gen.add_argument("--media-root", required=True, help="Directory the benchmark video paths are relative to")
    gen.add_argument("--model", required=True, help="DynThinker checkpoint directory")
    gen.add_argument("--output", required=True, help="Predictions JSONL (resumed only by an identical run)")
    gen.add_argument("--geometry-encoder-path", default=None)
    gen.add_argument("--num-frames", type=int, default=32, help="Uniformly sampled frames (research evaluation scripts: 32)")
    gen.add_argument("--max-new-tokens", type=int, default=None, help="Default: benchmark-specific")
    gen.add_argument("--device", default="cuda")
    gen.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    gen.add_argument("--attn-implementation", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    gen.add_argument("--max-pixels", type=int, default=451584)
    gen.add_argument("--min-pixels", type=int, default=12544)
    gen.add_argument("--limit", type=int, default=None, help="Only the first N samples")
    gen.set_defaults(func=cmd_generate)

    sc = sub.add_parser("score", help="Score an existing predictions JSONL")
    sc.add_argument("--benchmark", required=True, choices=sorted(BENCHMARKS))
    sc.add_argument("--predictions", required=True)
    sc.add_argument("--output", default=None, help="Optional JSON summary path")
    sc.set_defaults(func=cmd_score)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
