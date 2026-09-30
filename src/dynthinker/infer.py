# Copyright (c) 2026 The DynThinker Authors. Licensed under the MIT License.
"""`dynthinker-infer`: answer a question about a video or an ordered list of frames."""

import argparse
import json
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dynthinker-infer",
        description="Run a DynThinker (Qwen3-VL + VDPM + SDF) checkpoint on a video or image sequence.",
    )
    parser.add_argument("--model", required=True, help="DynThinker checkpoint directory (HF format)")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", help="Video file (.mp4/.avi/.mov/...); frames are sampled uniformly")
    source.add_argument("--images", nargs="+", help="Ordered frame images")
    parser.add_argument("--question", required=True, help="Question text (include answer-format instructions)")
    parser.add_argument("--num-frames", type=int, default=32, help="Frames sampled from --video (default: 32)")
    parser.add_argument(
        "--geometry-encoder-path",
        default=None,
        help="VDPM model.pt; required if the checkpoint was saved without VDPM weights "
        "(env DYNTHINKER_VDPM_PATH also works)",
    )
    parser.add_argument("--device", default="cuda", help="cuda, cuda:N or cpu (default: cuda)")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--attn-implementation", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0, help="0 = greedy decoding")
    parser.add_argument("--max-pixels", type=int, default=451584, help="Default matches the research evaluation scripts")
    parser.add_argument("--min-pixels", type=int, default=12544)
    parser.add_argument("--no-system-prompt", action="store_true", help="Omit the default system prompt")
    parser.add_argument("--json", action="store_true", help="Print a JSON object instead of plain text")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    from .inference import DynThinkerPipeline, load_frames

    frames = load_frames(video=args.video, images=args.images, num_frames=args.num_frames)
    pipeline = DynThinkerPipeline(
        args.model,
        geometry_encoder_path=args.geometry_encoder_path,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    answer = pipeline.generate(
        frames,
        args.question,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        add_system_prompt=not args.no_system_prompt,
    )
    if args.json:
        print(json.dumps({"question": args.question, "num_frames": len(frames), "answer": answer}))
    else:
        print(answer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
