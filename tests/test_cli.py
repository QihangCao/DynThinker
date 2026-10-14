"""Every console entry point parses --help without a GPU or weights."""

import os
import subprocess
import sys

import pytest

SRC = os.path.join(os.path.dirname(__file__), os.pardir, "src")
MODULES = [
    "dynthinker.train.train",
    "dynthinker.infer",
    "dynthinker.eval.run",
    "dynthinker.train.export_ema",
    "dynthinker.tools.ema_decay",
]


@pytest.mark.parametrize("module", MODULES)
def test_help(module):
    env = dict(os.environ, PYTHONPATH=SRC + os.pathsep + os.environ.get("PYTHONPATH", ""), PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-m", module, "--help"], capture_output=True, text=True, env=env, timeout=300)
    assert result.returncode == 0, result.stderr
    assert "usage" in result.stdout.lower()


def test_train_rejects_missing_dataset_config(tmp_path):
    from dynthinker.train.arguments import DataArguments, ModelArguments, TrainingArguments
    from transformers import HfArgumentParser

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(
        ["--output_dir", str(tmp_path), "--ema_enable", "True", "--ema_mlp_decay", "0.99", "--report_to", "none"]
    )
    assert model_args.geo_inject_version == "v1" and model_args.ema_mlp_layer_threshold == 4
    assert data_args.dataset_config is None and training_args.auto_resume
