from __future__ import annotations

import json
from pathlib import Path

import pytest


TRACE_MODEL_FIXTURE = Path("output/tinyllama-hybrid-ultra-test")


def _require_trace_model_fixture() -> Path:
    if not TRACE_MODEL_FIXTURE.exists():
        pytest.skip(f"optional trace model fixture not found: {TRACE_MODEL_FIXTURE}")
    return TRACE_MODEL_FIXTURE


def test_trace_cli_outputs_token_global_index_and_text(tmp_path: Path) -> None:
    # This test loads the repo's tinyllama minimal-weights directory to ensure the trace includes token-level fields.
    from vitriol.cli.commands.trace import trace as trace_cmd
    from click.testing import CliRunner

    model_path = _require_trace_model_fixture()
    out = tmp_path / "trace.json"

    runner = CliRunner()
    result = runner.invoke(
        trace_cmd,
        [
            "--model-path",
            str(model_path),
            "--prompt",
            "hello",
            "--max-new-tokens",
            "1",
            "--out",
            str(out),
            "--device",
            "cpu",
            "--trust-remote-code",
        ],
    )
    assert result.exit_code == 0, result.output

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["schema_version"] == "trace.v1"
    assert data["events"], "events should not be empty"
    e0 = data["events"][0]
    assert "token_global_index" in e0 and isinstance(e0["token_global_index"], int)
    assert "token_text" in e0 and isinstance(e0["token_text"], str)
    assert "attention_topk" in e0 and isinstance(e0["attention_topk"], dict)
    assert "attention_histogram" in e0 and isinstance(e0["attention_histogram"], dict)
    # Histogram must be emitted per-layer (key=block:{i}:attn).
    assert "block:0:attn" in e0["attention_histogram"]
    assert isinstance(e0["attention_histogram"]["block:0:attn"], dict)


def test_trace_cli_outputs_fine_grained_nodes(tmp_path: Path) -> None:
    """
    Fine-grained accuracy requirement:
    node_path should include key submodules inside a Llama block (norm/qkv/o_proj/gate/up/down),
    otherwise playback is only a coarse-grained demo.
    """
    from vitriol.cli.commands.trace import trace as trace_cmd
    from click.testing import CliRunner

    model_path = _require_trace_model_fixture()
    out = tmp_path / "trace.json"
    runner = CliRunner()
    result = runner.invoke(
        trace_cmd,
        [
            "--model-path",
            str(model_path),
            "--prompt",
            "hello",
            "--max-new-tokens",
            "1",
            "--out",
            str(out),
            "--device",
            "cpu",
            "--trust-remote-code",
        ],
    )
    assert result.exit_code == 0, result.output

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["events"], "events should not be empty"

    # It is sufficient for these to appear in any event (prefill or decode).
    all_paths: list[str] = []
    for ev in data["events"]:
        p = ev.get("node_path", [])
        if isinstance(p, list):
            all_paths.extend([str(x) for x in p])

    must_have = [
        "block:0:norm1",
        "block:0:attn:q_proj",
        "block:0:attn:k_proj",
        "block:0:attn:v_proj",
        "block:0:attn:o_proj",
        "block:0:norm2",
        "block:0:ffn:gate_proj",
        "block:0:ffn:up_proj",
        "block:0:ffn:down_proj",
    ]
    missing = [x for x in must_have if x not in all_paths]
    assert not missing, f"missing fine nodes: {missing}"
