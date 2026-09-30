# Copyright (c) 2026 The DynThinker Authors. Licensed under the MIT License.
"""Decoupled EMA (D-EMA) as a HuggingFace ``TrainerCallback``.

Trainable parameters are split into three groups (names follow ``model.named_parameters()``):

* ``mlp``        -- ``*.language_model.layers.{i}.mlp.*`` with ``i >= mlp_layer_threshold``, where ``i`` is
                    the 0-based decoder layer index. With the default ``threshold=4``, layers 4, 5, ...
                    (0-based), or layers 5 onward (1-based), form the sensitive group.
* ``new_module`` -- names containing ``cross_attn`` (SDF) or ``geometry_merger``.
* ``other``      -- every other trainable parameter (attention, early MLPs, norms, embeddings, lm_head, ...).

Updates happen after the optimizer steps ``s >= 1`` with ``s >= start_step`` and
``(s - start_step) % update_interval == 0``; each shadow is updated as
``shadow = beta * shadow + (1 - beta) * online`` with the group's ``beta``. Step 0 is not an optimizer
step, so ``start_step=0`` behaves like a schedule whose first update is at ``update_interval``.
Shadows are initialised from the parameters at train begin (the pretrained weights for pretrained groups).

The online model is optimised normally. When training finishes, the shadows of the groups whose
``*_save_as_final`` flag is set are copied into the model before the final save. With
``new_module_save_as_final=False`` (default) the new modules keep their online values, i.e. they are
not averaged (rho_new = 0).

With ``checkpoint_save=True`` the shadows are written next to every Trainer checkpoint as
``ema_state.safetensors`` + ``ema_state.json`` and restored on resume. The checkpoint itself keeps the
*online* weights, so resuming continues the unaveraged trajectory. The file layout and metadata match the
research code base, so EMA states written by it can be resumed as long as the D-EMA configuration is
identical. Resuming (``global_step > 0`` at train begin) without restorable EMA state is an error unless
``reinit_on_resume=True``: the shadows would restart from the resumed online weights instead of the
pretrained ones.
"""

import json
import logging
import os
import pathlib
import re
from typing import Dict, Iterable, List, Tuple

import torch
from transformers import TrainerCallback

logger = logging.getLogger(__name__)

EMA_STATE_FILENAME = "ema_state.safetensors"
EMA_META_FILENAME = "ema_state.json"
EMA_STATE_VERSION = 1
MAX_DECAY = 0.999999

_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
_LAYER_MLP_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp\.")


def extract_layer_idx(name: str):
    parts = name.split(".")
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return None


def is_late_mlp(name: str, threshold: int) -> bool:
    """``language_model.layers.[idx >= threshold].mlp.*``"""
    if ".language_model.layers." not in name or ".mlp." not in name:
        return False
    idx = extract_layer_idx(name)
    return idx is not None and idx >= threshold


def is_new_module(name: str) -> bool:
    """Newly introduced modules: SDF (``cross_attn``) and the geometry merger."""
    return "cross_attn" in name or "geometry_merger" in name


def group_parameter_names(
    named_parameters: Iterable[Tuple[str, torch.nn.Parameter]],
    mlp_layer_threshold: int,
    trainable_only: bool = True,
) -> Dict[str, List[str]]:
    """Assign parameter names to the ``mlp`` / ``other`` / ``new_module`` groups (in model order)."""
    groups = {"mlp": [], "other": [], "new_module": []}
    for name, param in named_parameters:
        if param is None or (trainable_only and not param.requires_grad):
            continue
        if is_late_mlp(name, mlp_layer_threshold):
            groups["mlp"].append(name)
        elif is_new_module(name):
            groups["new_module"].append(name)
        else:
            groups["other"].append(name)
    return groups


def first_update_step(start_step: int, update_interval: int) -> int:
    """First optimizer step (>= 1) that triggers an EMA update."""
    return start_step if start_step >= 1 else update_interval


def is_update_step(step: int, start_step: int, update_interval: int) -> bool:
    return step >= 1 and step >= start_step and (step - start_step) % update_interval == 0


def expected_update_count(step: int, start_step: int, update_interval: int) -> int:
    """Number of EMA updates performed after ``step`` optimizer steps (steps 1..step)."""
    first = first_update_step(start_step, update_interval)
    if step < first:
        return 0
    return ((step - first) // update_interval) + 1


class DecoupledEMACallback(TrainerCallback):
    """Group-wise EMA of trainable parameters (see module docstring)."""

    def __init__(
        self,
        mlp_decay: float,
        other_decay: float = -1.0,
        new_module_decay: float = -1.0,
        mlp_layer_threshold: int = 4,
        start_step: int = 200,
        update_interval: int = 1,
        trainable_only: bool = True,
        mlp_save_as_final: bool = True,
        other_save_as_final: bool = True,
        new_module_save_as_final: bool = False,
        decay: float = 0.999,
        on_gpu: bool = False,
        rank_zero_only: bool = False,
        checkpoint_save: bool = False,
        checkpoint_strict: bool = True,
        checkpoint_dtype: str = "float32",
        reinit_on_resume: bool = False,
    ):
        if not 0.0 <= float(mlp_decay) < 1.0:
            raise ValueError(f"D-EMA requires 0 <= ema_mlp_decay < 1, got {mlp_decay}")
        self.decay = float(decay)
        self.start_step = max(0, int(start_step))
        self.update_interval = max(1, int(update_interval))
        self.trainable_only = bool(trainable_only)
        self.mlp_layer_threshold = int(mlp_layer_threshold)
        self.mlp_decay = float(mlp_decay)
        self.other_decay = float(other_decay)
        self.new_module_decay = float(new_module_decay)
        self.mlp_save_as_final = bool(mlp_save_as_final)
        self.other_save_as_final = bool(other_save_as_final)
        self.new_module_save_as_final = bool(new_module_save_as_final)
        self.on_gpu = bool(on_gpu)
        self.rank_zero_only = bool(rank_zero_only)
        self.checkpoint_save = bool(checkpoint_save)
        self.checkpoint_strict = bool(checkpoint_strict)
        self.checkpoint_dtype = str(checkpoint_dtype).lower()
        self.reinit_on_resume = bool(reinit_on_resume)
        if self.checkpoint_dtype not in _DTYPES:
            raise ValueError(
                f"ema_checkpoint_dtype must be one of {sorted(_DTYPES)}; got {checkpoint_dtype!r}"
            )

        self.param_names: Dict[str, List[str]] = {"mlp": [], "other": [], "new_module": []}
        self.shadows: Dict[str, Dict[str, torch.Tensor]] = {"mlp": {}, "other": {}, "new_module": {}}
        self._pending_log = None
        self._resume_checkpoint = None
        self.num_updates = 0
        self.last_ema_step = None
        self._is_ema_owner = True

    # ------------------------------------------------------------------ decays
    def group_decays(self) -> Dict[str, float]:
        """Effective per-group decay (``<0`` falls back: other -> decay, new_module -> other)."""
        mlp = min(MAX_DECAY, max(0.0, self.mlp_decay))
        other = min(MAX_DECAY, max(0.0, self.other_decay if self.other_decay >= 0.0 else self.decay))
        new = min(MAX_DECAY, max(0.0, self.new_module_decay if self.new_module_decay >= 0.0 else other))
        return {"mlp": mlp, "other": other, "new_module": new}

    def _checkpoint_config(self) -> dict:
        # Keys and legacy constants mirror the research code's metadata so that its EMA states resume.
        return {
            "decay": self.decay,
            "start_step": self.start_step,
            "update_interval": self.update_interval,
            "trainable_only": self.trainable_only,
            "split_mode": True,
            "mlp_layer_threshold": self.mlp_layer_threshold,
            "mlp_decay": self.mlp_decay,
            "other_decay": self.other_decay,
            "mlp_save_as_final": self.mlp_save_as_final,
            "other_save_as_final": self.other_save_as_final,
            "new_module_decay": self.new_module_decay,
            "new_module_save_as_final": self.new_module_save_as_final,
            "dynamic_enable": False,
            "dynamic_groups": ["mlp"],
            "dynamic_start_decay": 0.99,
            "dynamic_end_step": 500,
            "dynamic_schedule": "cosine",
        }

    # ------------------------------------------------------------ checkpoints
    def set_resume_checkpoint(self, checkpoint_path):
        self._resume_checkpoint = str(checkpoint_path) if checkpoint_path is not None else None

    @staticmethod
    def has_checkpoint_state(checkpoint_path) -> bool:
        checkpoint_path = pathlib.Path(checkpoint_path)
        return (checkpoint_path / EMA_STATE_FILENAME).is_file() and (checkpoint_path / EMA_META_FILENAME).is_file()

    def save_checkpoint_state(self, checkpoint_dir: str, global_step: int) -> None:
        from safetensors.torch import save_file

        checkpoint_path = pathlib.Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        state_path = checkpoint_path / EMA_STATE_FILENAME
        meta_path = checkpoint_path / EMA_META_FILENAME
        state_tmp = checkpoint_path / f".{EMA_STATE_FILENAME}.tmp"
        meta_tmp = checkpoint_path / f".{EMA_META_FILENAME}.tmp"

        expected = expected_update_count(global_step, self.start_step, self.update_interval)
        if self.num_updates != expected:
            raise RuntimeError(
                "Refusing to checkpoint inconsistent EMA state: "
                f"global_step={global_step}, updates={self.num_updates}, expected_updates={expected}"
            )

        storage_dtype = _DTYPES[self.checkpoint_dtype]
        tensors = {}
        group_names = {}
        for group_name, shadow in self.shadows.items():
            group_names[group_name] = list(shadow.keys())
            for name, value in shadow.items():
                tensor = value.detach()
                if tensor.device.type != "cpu" or tensor.dtype != storage_dtype:
                    tensor = tensor.to(device="cpu", dtype=storage_dtype)
                tensors[f"{group_name}:{name}"] = tensor.contiguous()

        for tmp_path in (state_tmp, meta_tmp):
            tmp_path.unlink(missing_ok=True)
        try:
            save_file(tensors, str(state_tmp))
            metadata = {
                "version": EMA_STATE_VERSION,
                "global_step": int(global_step),
                "last_ema_step": self.last_ema_step,
                "num_updates": int(self.num_updates),
                "checkpoint_dtype": self.checkpoint_dtype,
                "tensor_count": len(tensors),
                "state_file_size": state_tmp.stat().st_size,
                "groups": group_names,
                "config": self._checkpoint_config(),
            }
            with meta_tmp.open("w", encoding="utf-8") as handle:
                json.dump(metadata, handle, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(state_tmp, state_path)
            os.replace(meta_tmp, meta_path)
        finally:
            state_tmp.unlink(missing_ok=True)
            meta_tmp.unlink(missing_ok=True)

        logger.info(
            "[D-EMA] checkpoint saved step=%d updates=%d tensors=%d dtype=%s size=%.2f GiB path=%s",
            global_step, self.num_updates, len(tensors), self.checkpoint_dtype,
            state_path.stat().st_size / (1024**3), state_path,
        )

    def load_checkpoint_state(self, checkpoint_dir: str, global_step: int) -> bool:
        from safetensors import safe_open

        checkpoint_path = pathlib.Path(checkpoint_dir)
        state_path = checkpoint_path / EMA_STATE_FILENAME
        meta_path = checkpoint_path / EMA_META_FILENAME
        if not state_path.is_file() or not meta_path.is_file():
            message = (
                f"EMA checkpoint state is missing in {checkpoint_path}; expected "
                f"{EMA_STATE_FILENAME} and {EMA_META_FILENAME}"
            )
            if self.checkpoint_strict:
                raise RuntimeError(message)
            logger.warning("[D-EMA] %s; reinitializing shadow from the resumed model", message)
            return False

        with meta_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if metadata.get("version") != EMA_STATE_VERSION:
            raise RuntimeError(f"Unsupported EMA checkpoint version {metadata.get('version')} in {meta_path}")
        if int(metadata.get("global_step", -1)) != int(global_step):
            raise RuntimeError(
                f"EMA checkpoint step mismatch: state={metadata.get('global_step')}, "
                f"trainer={global_step}, path={checkpoint_path}"
            )
        if metadata.get("config") != self._checkpoint_config():
            raise RuntimeError(f"EMA configuration mismatch while resuming from {checkpoint_path}")
        if int(metadata.get("state_file_size", -1)) != state_path.stat().st_size:
            raise RuntimeError(f"EMA checkpoint file size mismatch: {state_path}")
        expected_groups = {name: list(values.keys()) for name, values in self.shadows.items()}
        if metadata.get("groups") != expected_groups:
            raise RuntimeError(f"EMA parameter groups do not match the resumed model in {checkpoint_path}")

        expected_keys = {f"{group}:{name}" for group, shadow in self.shadows.items() for name in shadow}
        with safe_open(str(state_path), framework="pt", device="cpu") as reader:
            actual_keys = set(reader.keys())
            if actual_keys != expected_keys:
                missing = sorted(expected_keys - actual_keys)[:5]
                unexpected = sorted(actual_keys - expected_keys)[:5]
                raise RuntimeError(
                    f"EMA tensor keys do not match the resumed model: missing={missing}, unexpected={unexpected}"
                )
            for group_name, shadow in self.shadows.items():
                for name, target in shadow.items():
                    source = reader.get_tensor(f"{group_name}:{name}")
                    if tuple(source.shape) != tuple(target.shape):
                        raise RuntimeError(
                            f"EMA tensor shape mismatch for {name}: "
                            f"checkpoint={tuple(source.shape)}, model={tuple(target.shape)}"
                        )
                    target.copy_(source.to(device=target.device, dtype=target.dtype))

        self.num_updates = int(metadata.get("num_updates", -1))
        self.last_ema_step = metadata.get("last_ema_step")
        expected = expected_update_count(global_step, self.start_step, self.update_interval)
        if self.num_updates != expected:
            raise RuntimeError(
                "EMA update-count mismatch while resuming: "
                f"checkpoint={self.num_updates}, expected={expected}, step={global_step}"
            )
        logger.info(
            "[D-EMA] checkpoint restored step=%d updates=%d last_ema_step=%s path=%s",
            global_step, self.num_updates, self.last_ema_step, state_path,
        )
        return True

    # ------------------------------------------------------------- lifecycle
    def _make_shadow(self, param: torch.nn.Parameter) -> torch.Tensor:
        tensor = param.detach().float()
        if not self.on_gpu:
            tensor = tensor.cpu()
        return tensor.clone()

    def init_shadows(self, model: torch.nn.Module) -> None:
        real_model = model.module if hasattr(model, "module") else model
        named = list(real_model.named_parameters())
        self.param_names = group_parameter_names(named, self.mlp_layer_threshold, self.trainable_only)

        if not self.param_names["mlp"]:
            # Guard against backbones whose decoder layers are not named `language_model.layers.*`:
            # a silently empty sensitive group would turn D-EMA into a single shared decay.
            unmatched = [
                name
                for name, param in named
                if (param.requires_grad or not self.trainable_only)
                and (match := _LAYER_MLP_PATTERN.search(name))
                and int(match.group(1)) >= self.mlp_layer_threshold
            ]
            if unmatched:
                raise RuntimeError(
                    "D-EMA found late-layer MLP parameters that do not follow the "
                    f"'*.language_model.layers.<i>.mlp.*' naming (e.g. {unmatched[0]!r}); "
                    "the sensitive group would be empty."
                )

        params = dict(named)
        self.shadows = {
            group: {name: self._make_shadow(params[name]) for name in names}
            for group, names in self.param_names.items()
        }
        decays = self.group_decays()
        logger.info(
            "[D-EMA] groups: mlp=%d (decay %s, layer >= %d), other=%d (decay %s), new_module=%d (decay %s); "
            "save_as_final mlp=%s other=%s new_module=%s",
            len(self.param_names["mlp"]), decays["mlp"], self.mlp_layer_threshold,
            len(self.param_names["other"]), decays["other"],
            len(self.param_names["new_module"]), decays["new_module"],
            self.mlp_save_as_final, self.other_save_as_final, self.new_module_save_as_final,
        )

    def planned_drift(self, max_steps: int) -> Tuple[int, Dict[str, float]]:
        """Updates in a run of ``max_steps`` optimizer steps and each group's drift ``1 - decay^updates``."""
        updates = expected_update_count(max_steps, self.start_step, self.update_interval)
        return updates, {group: 1.0 - decay**updates for group, decay in self.group_decays().items()}

    def log_planned_drift(self, max_steps: int) -> None:
        if max_steps <= 0:
            return
        updates, drift = self.planned_drift(max_steps)
        logger.info(
            "[D-EMA] max_steps=%d -> %d updates; realised drift mlp=%.4f other=%.4f new_module=%.4f",
            max_steps, updates, drift["mlp"], drift["other"], drift["new_module"],
        )

    def on_train_begin(self, args, state, control, **kwargs):
        self._is_ema_owner = not self.rank_zero_only or bool(getattr(state, "is_world_process_zero", True))
        if not self._is_ema_owner:
            return control
        model = kwargs.get("model")
        if model is None:
            return control
        self.init_shadows(model)
        self.log_planned_drift(int(getattr(state, "max_steps", 0) or 0))
        step = int(state.global_step)
        restored = False
        if self.checkpoint_save and self._resume_checkpoint is not None:
            restored = self.load_checkpoint_state(self._resume_checkpoint, step)
        if step > 0 and not restored:
            # checkpoint_strict=False already opts in to re-initialisation when the saved state is missing.
            if not (self.reinit_on_resume or (self.checkpoint_save and not self.checkpoint_strict)):
                raise RuntimeError(
                    f"[D-EMA] training resumes at step {step} but no EMA state is restored "
                    f"(checkpoint_save={self.checkpoint_save}). The shadows would restart from the resumed "
                    "weights, not the pretrained ones. Train with --ema_checkpoint_save True, or pass "
                    "--ema_reinit_on_resume True to accept re-initialisation."
                )
            # The update counter follows the schedule so that later checkpoints stay resumable.
            self.num_updates = expected_update_count(step, self.start_step, self.update_interval)
            logger.warning(
                "[D-EMA] resuming at step %d without EMA state: shadows re-initialised from the resumed weights", step
            )
        return control

    def _update_group(self, group: str, named_params, beta: float) -> None:
        alpha = 1.0 - beta
        shadow = self.shadows[group]
        for name in self.param_names[group]:
            param = named_params.get(name)
            ema_value = shadow.get(name)
            if param is None or ema_value is None:
                continue
            current = param.detach().float() if self.on_gpu else param.detach().float().cpu()
            ema_value.mul_(beta).add_(current, alpha=alpha)

    def update(self, model: torch.nn.Module, step: int) -> None:
        real_model = model.module if hasattr(model, "module") else model
        named_params = dict(real_model.named_parameters())
        decays = self.group_decays()
        with torch.no_grad():
            for group in ("mlp", "other", "new_module"):
                self._update_group(group, named_params, decays[group])
        self.num_updates += 1
        self.last_ema_step = step
        self._pending_log = {
            "ema_mlp_decay": decays["mlp"],
            "ema_other_decay": decays["other"],
            "ema_new_module_decay": decays["new_module"],
            "ema_step": float(step),
        }

    def on_step_end(self, args, state, control, **kwargs):
        if not self._is_ema_owner or not any(self.param_names.values()):
            return control
        step = int(state.global_step)
        if not is_update_step(step, self.start_step, self.update_interval):
            return control
        model = kwargs.get("model")
        if model is None:
            return control
        self.update(model, step)
        return control

    def on_save(self, args, state, control, **kwargs):
        if not self.checkpoint_save:
            return control
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{int(state.global_step)}")
        if bool(state.is_world_process_zero):
            self.save_checkpoint_state(checkpoint_dir, int(state.global_step))
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and self._pending_log is not None:
            logs.update(self._pending_log)
            self._pending_log = None
        return control

    # ---------------------------------------------------------------- export
    def final_groups(self) -> List[str]:
        flags = {
            "mlp": self.mlp_save_as_final,
            "other": self.other_save_as_final,
            "new_module": self.new_module_save_as_final,
        }
        return [group for group, flag in flags.items() if flag]

    def maybe_apply_to_model(self, model: torch.nn.Module) -> int:
        """Copy the shadows of the ``*_save_as_final`` groups into ``model``; returns #tensors."""
        if not self._is_ema_owner:
            return 0
        real_model = model.module if hasattr(model, "module") else model
        named_params = dict(real_model.named_parameters())
        applied = 0
        with torch.no_grad():
            for group in self.final_groups():
                for name in self.param_names[group]:
                    param = named_params.get(name)
                    ema_value = self.shadows[group].get(name)
                    if param is None or ema_value is None:
                        continue
                    param.copy_(ema_value.to(device=param.device, dtype=param.dtype))
                    applied += 1
        return applied


# Name used by the research code base.
OnlineEMACallback = DecoupledEMACallback
