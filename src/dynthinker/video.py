# Adopted from lmms-eval (https://github.com/EvolvingLMMs-Lab/lmms-eval,
# lmms_eval/models/model_utils/load_video.py). Below is the original license notice:
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
# Reduced to the uniform sampling protocol used for DynThinker evaluation.
"""Uniform video frame sampling with PyAV."""

from typing import List

import numpy as np
from PIL import Image


def uniform_indices(total_frames: int, num_frames: int) -> np.ndarray:
    """``num_frames`` indices spread uniformly over the clip (fewer if the clip is shorter)."""
    total_frames = int(total_frames)
    if total_frames <= 0:
        return np.asarray([], dtype=int)
    target = min(total_frames, int(num_frames))
    return np.linspace(0, total_frames - 1, target, dtype=int)


def _decode_selected(container, indices) -> list:
    indices = [int(index) for index in indices]
    wanted = set(indices)
    selected = {}
    end_index = max(indices)
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i in wanted:
            selected[i] = frame
    if any(index not in selected for index in indices):
        return []
    return [selected[index] for index in indices]


def read_video_frames(video_path: str, num_frames: int = 32) -> List[Image.Image]:
    """Decode ``num_frames`` uniformly sampled RGB frames as PIL images."""
    import av

    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    frames = []
    if not str(video_path).lower().endswith((".webm", ".mkv")):
        # Fast path using the declared frame count; falls back to full decoding if it is unreliable.
        try:
            with av.open(video_path) as container:
                indices = uniform_indices(container.streams.video[0].frames, num_frames)
                if len(indices):
                    frames = _decode_selected(container, indices)
        except Exception:
            frames = []
    if not frames:
        with av.open(video_path) as container:
            decoded = [frame for packet in container.demux(video=0) for frame in packet.decode()]
        indices = uniform_indices(len(decoded), num_frames)
        frames = [decoded[i] for i in indices]
    if not frames:
        raise ValueError(f"PyAV decoded no video frames from {video_path}")
    return [Image.fromarray(frame.to_ndarray(format="rgb24")) for frame in frames]
