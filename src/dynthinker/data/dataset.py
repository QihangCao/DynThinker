# Adapted from the Qwen-VL fine-tuning code (https://github.com/QwenLM/Qwen3-VL,
# qwen-vl-finetune/qwenvl/data/data_qwen.py, Apache License 2.0; Copyright Alibaba Cloud). Kept from
# that code: conversation tokenisation and loss masking (`preprocess_qwen_2_visual`), dataset
# sampling rates, retry-on-error loading, `pad_and_cat` and the core of the collator. Modified by the
# DynThinker authors: dataset registry, annotation loading, frame sampling, the per-frame VDPM/Qwen
# inputs, length estimates, visual-token compression and geometry-input collation.
"""Lazy supervised dataset: images or video frames + conversations -> Qwen-VL training tensors."""

import copy
import itertools
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import torch
import transformers
from PIL import Image
from torch.utils.data import Dataset

from .preprocess import prepare_image_inputs
from .registry import data_list, load_registry
from .rope2d import get_rope_index_2, get_rope_index_25

try:
    from decord import VideoReader
except ImportError:
    VideoReader = None

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"
SYSTEM_MESSAGE = "You are a helpful assistant."
CHAT_TEMPLATE = (
    "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + "
    "'<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


# Record fields of the research data that this release cannot honour; such records are rejected.
UNSUPPORTED_RECORD_FIELDS = {
    "spar_info": (
        "SPAR-7M marker annotations are not drawn by this release; render the markers into the images "
        "beforehand and remove the field, or leave the dataset out (see docs/DATA.md)"
    ),
}


def load_annotations(path: str, max_samples: int = -1) -> List[dict]:
    """Records of a .json (list) or .jsonl file; with ``max_samples >= 0`` only the first ones."""
    limit = None if max_samples == -1 else max_samples
    with open(path, "r", encoding="utf-8") as handle:
        if not path.endswith(".jsonl"):
            return json.load(handle)[:limit]
        rows = (json.loads(line) for line in handle if line.strip())
        return list(itertools.islice(rows, limit))


def check_record_fields(records, dataset_name: str) -> None:
    for field, reason in UNSUPPORTED_RECORD_FIELDS.items():
        count = sum(1 for record in records if record.get(field) is not None)
        if count:
            raise ValueError(f"dataset {dataset_name!r}: {count} record(s) use {field!r}. {reason}")


def _text(turn: dict) -> str:
    return turn.get("value", turn.get("content", ""))


def _set_text(turn: dict, text: str) -> None:
    turn["value" if "value" in turn else "content"] = text


def expand_video_token(conversations, num_frames: int) -> None:
    """Replace the single `<video>` placeholder by one `<image>` per sampled frame (in place)."""
    holders = [turn for turn in conversations if DEFAULT_VIDEO_TOKEN in _text(turn)]
    if len(holders) != 1:
        raise ValueError(f"expected exactly one turn with {DEFAULT_VIDEO_TOKEN}, found {len(holders)}")
    _set_text(holders[0], _text(holders[0]).replace(DEFAULT_VIDEO_TOKEN, DEFAULT_IMAGE_TOKEN * num_frames))


def preprocess_qwen_2_visual(sources, tokenizer, grid_thw: List = (), visual_type: str = "image") -> Dict:
    """Tokenize conversations; each `<image>` expands to `grid_thw[i]` pad tokens; only answers are supervised."""
    roles = {"human": "user", "gpt": "assistant", "user": "user", "assistant": "assistant"}
    system_message = SYSTEM_MESSAGE
    if visual_type not in ["image", "video"]:
        raise ValueError("visual_type must be either 'image' or 'video'")

    tokenizer = copy.deepcopy(tokenizer)
    tokenizer.chat_template = CHAT_TEMPLATE

    visual_replicate_index = 0
    input_ids, targets = [], []

    for source in sources:
        if source[0]["from"] == "system":
            system_message = source[0]["value"]
            source = source[1:]
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []
        input_id += tokenizer.apply_chat_template([{"role": "system", "content": system_message}])
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            role = conv.get("role", conv.get("from"))
            content = conv.get("content", conv.get("value"))
            role = roles.get(role, role)
            if role == "user":
                visual_tag = f"<{visual_type}>"
                if visual_tag in content:
                    parts = content.split(visual_tag)
                    new_parts = []
                    for j in range(len(parts) - 1):
                        new_parts.append(parts[j])
                        new_parts.append(
                            "<|vision_start|>"
                            + f"<|{visual_type}_pad|>" * grid_thw[visual_replicate_index]
                            + "<|vision_end|>"
                        )
                        visual_replicate_index += 1
                    new_parts.append(parts[-1])
                    content = "".join(new_parts)

            encode_id = tokenizer.apply_chat_template([{"role": role, "content": content}])
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target_mask = encode_id.copy()
                target_mask[:3] = [IGNORE_INDEX] * 3  # "<|im_start|>assistant\n"
                target += target_mask

        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        input_ids.append(input_id)
        targets.append(target)

    del tokenizer
    return dict(
        input_ids=torch.tensor(input_ids, dtype=torch.long),
        labels=torch.tensor(targets, dtype=torch.long),
    )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning.

    Every visual sample becomes a list of frames (an image list, or frames sampled from a video file or
    a directory of frame images). Each frame yields Qwen-VL patches and a VDPM input tensor.
    """

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_args):
        super().__init__()
        if not data_args.dataset_config:
            raise ValueError("--dataset_config is required (see configs/datasets.example.yaml)")
        registry = load_registry(data_args.dataset_config)
        dataset_list = data_list(data_args.dataset_use.split(","), registry)
        if not dataset_list:
            raise ValueError("--dataset_use selects no dataset")
        logger.info("Loading datasets: %s", [d["dataset_name"] for d in dataset_list])

        self.model_type = data_args.model_type
        self.get_rope_index = get_rope_index_25 if data_args.model_type == "qwen2.5vl" else get_rope_index_2

        list_data_dict = []
        for data in dataset_list:
            annotations = load_annotations(data["annotation_path"], data_args.max_samples)
            check_record_fields(annotations, data["dataset_name"])
            sampling_rate = data.get("sampling_rate", 1.0)
            if sampling_rate < 1.0:
                annotations = random.sample(annotations, int(len(annotations) * sampling_rate))
            logger.info("dataset %s: %d samples", data["dataset_name"], len(annotations))
            for record in annotations:
                record.update(data_path=data["data_path"], tag=data["tag"])
            list_data_dict.extend(annotations)

        logger.info("Total training samples: %d", len(list_data_dict))
        if getattr(data_args, "shuffle", True):
            random.shuffle(list_data_dict)

        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.data_args.image_processor.max_pixels = data_args.max_pixels
        self.data_args.image_processor.min_pixels = data_args.min_pixels
        self.data_args.image_processor.size["longest_edge"] = data_args.max_pixels
        self.data_args.image_processor.size["shortest_edge"] = data_args.min_pixels

        self.depart_smi_token = getattr(self.data_args, "depart_smi_token", False)
        self.smi_image_num = getattr(self.data_args, "smi_image_num", 8)
        self.smi_downsample_rate = getattr(self.data_args, "smi_downsample_rate", 2)
        self.cached_lengths = [self._estimate_length(sample) for sample in self.list_data_dict]

    # ---------------------------------------------------------------- lengths
    def _compressed(self, num_frames: int) -> bool:
        return bool(self.depart_smi_token) and num_frames > self.smi_image_num

    def _estimate_length(self, sample) -> int:
        """Whitespace words + 252 tokens per frame (divided by rate^2 when frames are compressed)."""
        words = sum(len(_text(turn).split()) for turn in sample["conversations"])
        listed = next((sample[key] for key in ("image", "images") if key in sample), None)
        if listed is not None:
            frames = len(listed)
        else:
            frames = getattr(self.data_args, "video_max_frames", 8) if "video" in sample else 0
        visual_tokens = frames * 252
        if self._compressed(frames):
            visual_tokens //= self.smi_downsample_rate**2
        return visual_tokens + words

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        return self.cached_lengths

    @property
    def modality_lengths(self):
        """Signed lengths for the modality-grouped sampler: negative marks "2d" samples."""
        return [
            -length if sample.get("tag", "2d") == "2d" else length
            for sample, length in zip(self.list_data_dict, self.cached_lengths)
        ]

    # ------------------------------------------------------------------ media
    def frame_indices(self, total_frames: int, fps: float = 1.0) -> np.ndarray:
        """Sorted unique indices of evenly spaced frames: one per `base_interval` seconds of footage,
        at least `video_min_frames` and at most `video_max_frames` of them."""
        seconds_per_frame = getattr(self.data_args, "base_interval", 2)
        low = getattr(self.data_args, "video_min_frames", 4)
        high = getattr(self.data_args, "video_max_frames", 8)
        count = min(high, max(low, round(total_frames / fps / seconds_per_frame)))
        return np.unique(np.linspace(0, total_frames - 1, count, dtype=int))

    @staticmethod
    def _decode_video(video_file, frame_index_fn):
        """Decode the selected RGB frames with Decord if available, otherwise PyAV."""
        if VideoReader is not None:
            reader = VideoReader(video_file, num_threads=4)
            try:
                frame_idx = frame_index_fn(len(reader), float(reader.get_avg_fps()))
                return reader.get_batch(frame_idx).asnumpy()
            finally:
                del reader

        import av

        with av.open(video_file) as container:
            if not container.streams.video:
                raise ValueError(f"No video stream found in {video_file}")
            stream = container.streams.video[0]
            rate = stream.average_rate or stream.base_rate or stream.guessed_rate
            avg_fps = float(rate) if rate else 1.0
            declared_frames = int(stream.frames or 0)
            if declared_frames > 0:
                frame_idx = frame_index_fn(declared_frames, avg_fps)
                wanted = {int(index) for index in frame_idx}
                selected = {}
                for position, frame in enumerate(container.decode(stream)):
                    if position in wanted:
                        selected[position] = frame.to_ndarray(format="rgb24")
                    if len(selected) == len(wanted):
                        break
                if all(int(index) in selected for index in frame_idx):
                    return np.stack([selected[int(index)] for index in frame_idx])

        # The container does not expose a reliable frame count: decode once and sample.
        with av.open(video_file) as container:
            stream = container.streams.video[0]
            decoded = [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]
        if not decoded:
            raise ValueError(f"No decodable video frames found in {video_file}")
        frame_idx = frame_index_fn(len(decoded), avg_fps)
        return np.stack([decoded[int(index)] for index in frame_idx])

    @staticmethod
    def _open_rgb(path: str) -> Image.Image:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            return image.convert("RGB").copy()

    def load_video_frames(self, data_path: str, video: str) -> List[Image.Image]:
        """Frames sampled from a video file (.mp4/.avi/.mov) or a directory of frame images (1 fps)."""
        if not isinstance(video, str):
            raise ValueError("'video' must be a path string")
        location = os.path.join(data_path, video)
        if os.path.isdir(location):
            files = sorted(e.path for e in os.scandir(location) if e.is_file())
            return [self._open_rgb(files[k]) for k in self.frame_indices(len(files), 1)]
        if location.endswith((".mp4", ".avi", ".mov")) and os.path.exists(location):
            return [Image.fromarray(a).convert("RGB") for a in self._decode_video(location, self.frame_indices)]
        raise FileNotFoundError(location)

    def load_image_frames(self, data_path: str, images) -> List[Image.Image]:
        """An ordered image list (paths relative to `data_path`, or PIL images), all at the first one's size."""
        if not isinstance(images, list) or not images:
            raise ValueError("'image'/'images' must be a non-empty list")
        frames = []
        for entry in images:
            if isinstance(entry, str):
                entry = self._open_rgb(os.path.join(data_path, entry))
            elif not isinstance(entry, Image.Image):
                raise TypeError(f"Unsupported image entry type: {type(entry)}")
            frames.append(entry)
        size = frames[0].size
        return [f if f.size == size else f.resize(size, Image.BILINEAR) for f in frames]

    # ---------------------------------------------------------------- samples
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        num_base_retries = 3
        for attempt_idx in range(num_base_retries):
            try:
                return self._get_item(i)
            except Exception as error:
                logger.warning("[Try #%d] Failed to fetch sample %d: %s", attempt_idx, i, error)
                time.sleep(1)

        # Try the next sample, in case the file is corrupted.
        next_index = min(i + 1, len(self.list_data_dict) - 1)
        for attempt_idx in range(num_base_retries):
            try:
                return self._get_item(next_index)
            except Exception as error:
                logger.warning("[Try other #%d] Failed to fetch sample %d: %s", attempt_idx, next_index, error)
        return self._get_item(i)

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        record = self.list_data_dict[i]
        conversations = copy.deepcopy(record["conversations"])
        tag = record.get("tag", "2d")

        # Precedence of the research data loader: "images", then "video", then "image".
        frames = None
        if "images" in record:
            frames = self.load_image_frames(record["data_path"], record["images"])
        elif "video" in record:
            frames = self.load_video_frames(record["data_path"], record["video"])
            expand_video_token(conversations, len(frames))
        elif "image" in record:
            frames = self.load_image_frames(record["data_path"], record["image"])
        for turn in conversations:
            _set_text(turn, _text(turn).replace(DEFAULT_IMAGE_TOKEN + "\n", DEFAULT_IMAGE_TOKEN))

        if frames is None:
            data_dict = preprocess_qwen_2_visual([conversations], self.tokenizer, grid_thw=None)
            position_ids = torch.arange(0, data_dict["input_ids"].size(1)).view(1, -1).unsqueeze(0).expand(3, -1, -1)
            return dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0], position_ids=position_ids, tag=tag)

        per_frame = [prepare_image_inputs(frame, self.data_args.image_processor) for frame in frames]
        grids = [item["image_grid_thw"] for item in per_frame]
        # Visual-token compression is decided from the number of <image> tokens in the first turn.
        merge = self.data_args.image_processor.merge_size
        if self._compressed(_text(conversations[0]).count(DEFAULT_IMAGE_TOKEN)):
            merge *= self.smi_downsample_rate
        tokens_per_frame = [int(t * (h // merge) * (w // merge)) for t, h, w in (g.tolist() for g in grids)]

        data_dict = preprocess_qwen_2_visual([conversations], self.tokenizer, grid_thw=tokens_per_frame, visual_type="image")
        position_ids, _ = self.get_rope_index(merge, data_dict["input_ids"], torch.stack(grids))
        item = dict(
            input_ids=data_dict["input_ids"][0],
            labels=data_dict["labels"][0],
            position_ids=position_ids,
            tag=tag,
            pixel_values=[x["pixel_values"] for x in per_frame],
            image_grid_thw=grids,
        )
        if getattr(self.data_args, "use_geometry_encoder", False):
            item["geometry_encoder_inputs"] = [x["geometry_encoder_inputs"] for x in per_frame]
        return item


def pad_and_cat(tensor_list):
    max_length = max(tensor.shape[2] for tensor in tensor_list)
    padded = [torch.nn.functional.pad(t, (0, max_length - t.shape[2]), "constant", 1) for t in tensor_list]
    return torch.cat(padded, dim=1)


@dataclass
class DataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels", "position_ids")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        position_ids = pad_and_cat(position_ids)
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        position_ids = position_ids[:, :, : self.tokenizer.model_max_length]
        batch = dict(input_ids=input_ids, labels=labels, attention_mask=input_ids.ne(self.tokenizer.pad_token_id))

        images = list(itertools.chain(*(instance["pixel_values"] for instance in instances if "pixel_values" in instance)))
        if images:
            batch["pixel_values"] = torch.cat(images, dim=0)
            grid_thw = list(
                itertools.chain(*(instance["image_grid_thw"] for instance in instances if "image_grid_thw" in instance))
            )
            batch["image_grid_thw"] = torch.stack(grid_thw, dim=0)
        else:
            batch["pixel_values"] = None
            batch["image_grid_thw"] = None
        batch["pixel_values_videos"] = None
        batch["video_grid_thw"] = None
        batch["position_ids"] = position_ids

        # Geometry batches: one [S, 3, H, W] frame stack per sample; the modality-grouped sampler keeps
        # "2d" and "3d" samples in separate batches, which is checked here.
        if "geometry_encoder_inputs" in instances[0]:
            tags = sorted({instance["tag"] for instance in instances})
            if len(tags) != 1:
                raise ValueError(f"a geometry batch must not mix dataset tags, got {tags}")
            batch["tag"] = tags[0]
            batch["geometry_encoder_inputs"] = [torch.stack(instance["geometry_encoder_inputs"]) for instance in instances]
        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_args=data_args)
    return dict(
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=DataCollatorForSupervisedDataset(tokenizer=tokenizer),
    )
