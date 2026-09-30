"""Benchmark prompting and scoring (VSI-Bench MRA / accuracy, VLM4D choice matching)."""

import json
import os

import pytest

from dynthinker.eval.benchmarks import (
    mean_relative_accuracy,
    vlm4d_prompt,
    vlm4d_score,
    vlm4d_video_path,
    vsibench_aggregate,
    vsibench_prompt,
    vsibench_score,
)
from dynthinker.eval.run import main


def test_mean_relative_accuracy():
    assert mean_relative_accuracy(2.0, 2.0) == 1.0
    # |1.5 - 2| / 2 = 0.25 <= 1 - c for c in {0.5, ..., 0.75} -> 6 of 10 thresholds
    assert mean_relative_accuracy(1.5, 2.0) == pytest.approx(0.6)
    assert mean_relative_accuracy(10.0, 2.0) == 0.0


def test_vsibench_prompt_and_score():
    mca = {"question_type": "route_planning", "question": "Q?", "options": ["A. left", "B. right"], "ground_truth": "B"}
    na = {"question_type": "object_counting", "question": "How many?", "ground_truth": "4"}
    assert vsibench_prompt(mca).endswith("Answer with the option's letter from the given choices directly.")
    assert "Options:\nA. left\nB. right" in vsibench_prompt(mca)
    assert vsibench_prompt(na) == "These are frames of a video.\nHow many?\nPlease answer the question using a single word or phrase."
    assert vsibench_score(mca, "B. right")["accuracy"] == 1.0
    assert vsibench_score(mca, "A")["accuracy"] == 0.0
    assert vsibench_score(na, "4")["MRA:.5:.95:.05"] == 1.0
    assert vsibench_score(na, "four")["MRA:.5:.95:.05"] == 0.0


def test_vsibench_aggregate_averages_direction_levels():
    records = []
    for level, acc in (("easy", 1.0), ("medium", 0.0), ("hard", 0.5)):
        records.append({"question_type": f"object_rel_direction_{level}", "score": {"accuracy": acc}})
    records.append({"question_type": "object_counting", "score": {"MRA:.5:.95:.05": 0.5}})
    metrics = vsibench_aggregate(records)
    assert metrics["object_rel_direction_accuracy"] == pytest.approx(50.0)
    assert metrics["overall"] == pytest.approx(50.0)


def test_vlm4d():
    doc = {"video": "https://huggingface.co/datasets/x/resolve/main/videos/a.mp4", "question": "Where?",
           "choices": "{'A': 'left', 'B': 'right'}", "answer": "right", "type": "ego"}
    assert vlm4d_video_path(doc, "/root") == "/root/videos/a.mp4"
    assert vlm4d_prompt(doc) == (
        "These are frames of a video.\nWhere?\nOptions:\nA. left\nB. right\n"
        "Answer with the option's letter from the given choices directly."
    )
    assert vlm4d_score(doc, "B")["accuracy"] == 1.0
    assert vlm4d_score(doc, "<answer>B</answer>")["accuracy"] == 1.0
    assert vlm4d_score(doc, "C")["accuracy"] == 0.0


def test_score_cli(tmp_path, capsys):
    preds = tmp_path / "preds.jsonl"
    docs = [
        {"question_type": "route_planning", "question": "q", "options": ["A. x", "B. y"], "ground_truth": "A"},
        {"question_type": "object_counting", "question": "q", "ground_truth": "3"},
    ]
    with preds.open("w") as handle:
        for i, (doc, pred) in enumerate(zip(docs, ["A", "3"])):
            handle.write(json.dumps({"index": i, "doc": doc, "prediction": pred}) + "\n")
    assert main(["score", "--benchmark", "vsibench", "--predictions", str(preds), "--output", str(tmp_path / "s.json")]) == 0
    summary = json.loads((tmp_path / "s.json").read_text())
    assert summary["num_samples"] == 2 and summary["metrics"]["overall"] == pytest.approx(100.0)


def _eval_setup(tmp_path, monkeypatch):
    """Two fake checkpoints, VSI-Bench style annotations and media; the pipeline is stubbed."""
    import dynthinker.inference as inference

    class StubPipeline:
        def __init__(self, model_path, **kwargs):
            self.name = os.path.basename(model_path)

        def generate(self, frames, prompt, max_new_tokens=16):
            return f"{self.name}:{len(frames)}"

    monkeypatch.setattr(inference, "DynThinkerPipeline", StubPipeline)
    monkeypatch.setattr(inference, "load_frames", lambda video, num_frames: [None] * num_frames)
    for name in ("model_a", "model_b"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "config.json").write_text(json.dumps({"name": name}))
        (tmp_path / name / "model.safetensors").write_bytes(name.encode() * 100)
    docs = [{"dataset": "d", "scene_name": f"s{i}", "question_type": "object_counting", "question": "q",
             "ground_truth": "3"} for i in range(3)]
    (tmp_path / "media" / "d").mkdir(parents=True)
    for doc in docs:
        (tmp_path / "media" / "d" / f"{doc['scene_name']}.mp4").write_bytes(b"")
    (tmp_path / "x.jsonl").write_text("".join(json.dumps(d) + "\n" for d in docs))
    (tmp_path / "y.jsonl").write_text("".join(json.dumps(d) + "\n" for d in reversed(docs)))

    def generate(model="model_a", annotations="x.jsonl", *extra):
        return main(["generate", "--benchmark", "vsibench", "--annotations", str(tmp_path / annotations),
                     "--media-root", str(tmp_path / "media"), "--model", str(tmp_path / model),
                     "--output", str(tmp_path / "preds.jsonl"), "--device", "cpu", *extra])

    return generate


def _lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_generate_resumes_only_the_same_run(tmp_path, monkeypatch):
    generate = _eval_setup(tmp_path, monkeypatch)
    preds = tmp_path / "preds.jsonl"
    assert generate("model_a", "x.jsonl", "--limit", "2") == 0
    first = _lines(preds)
    assert [r["index"] for r in first] == [0, 1] and len({r["run"] for r in first}) == 1
    assert json.loads((tmp_path / "preds.jsonl.run.json").read_text())["run"] == first[0]["run"]

    for args in (("model_b", "x.jsonl"), ("model_a", "y.jsonl"), ("model_a", "x.jsonl", "--num-frames", "8")):
        with pytest.raises(ValueError):
            generate(*args)
        assert _lines(preds) == first  # nothing appended

    assert generate("model_a", "x.jsonl") == 0  # same run, larger limit: resumes
    records = _lines(preds)
    assert [r["index"] for r in records] == [0, 1, 2]
    assert [r["doc"]["scene_name"] for r in records] == ["s0", "s1", "s2"]
    assert {r["prediction"] for r in records} == {"model_a:32"}


def test_score_rejects_duplicates_and_mixed_runs(tmp_path):
    doc = {"question_type": "object_counting", "question": "q", "ground_truth": "3"}
    preds = tmp_path / "p.jsonl"
    for records in (
        [{"index": 0, "run": "a", "doc": doc, "prediction": "3"}, {"index": 0, "run": "a", "doc": doc, "prediction": "3"}],
        [{"index": 0, "run": "a", "doc": doc, "prediction": "3"}, {"index": 1, "run": "b", "doc": doc, "prediction": "3"}],
    ):
        preds.write_text("".join(json.dumps(r) + "\n" for r in records))
        with pytest.raises(ValueError):
            main(["score", "--benchmark", "vsibench", "--predictions", str(preds)])


def test_generate_refuses_changed_legacy_chat_template(tmp_path, monkeypatch):
    generate = _eval_setup(tmp_path, monkeypatch)
    template = tmp_path / "model_a" / "chat_template.json"
    template.write_text(json.dumps({"chat_template": "original prompt"}))
    assert generate("model_a", "x.jsonl", "--limit", "1") == 0
    before = _lines(tmp_path / "preds.jsonl")
    template.write_text(json.dumps({"chat_template": "changed prompt"}))
    with pytest.raises(ValueError):
        generate("model_a", "x.jsonl")
    assert _lines(tmp_path / "preds.jsonl") == before


def test_malformed_predictions_error_names_file_and_line(tmp_path):
    from dynthinker.eval.run import _read_predictions
    path = tmp_path / "damaged.jsonl"
    path.write_text('{"index": 0}\n{"index":\n')
    with pytest.raises(ValueError) as raised:
        _read_predictions(str(path))
    message = str(raised.value)
    assert str(path) + ":2:" in message
    assert "--output" in message
