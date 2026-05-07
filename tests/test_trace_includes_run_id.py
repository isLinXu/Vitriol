from __future__ import annotations


def test_build_trace_v1_includes_run_id() -> None:
    from vitriol.cli.commands.trace import _build_trace_v1

    trace = _build_trace_v1(
        run_id="demo-run-id",
        model_path="output/demo-model",
        prompt="hello",
        max_new_tokens=1,
        prompt_token_ids=[1],
        prompt_tokens=["hello"],
        generated_token_ids=[2],
        generated_tokens=["!"],
        events=[{"token_index": 0, "phase": "prefill", "token_text": "hello", "token_global_index": 0, "attention_topk": {}, "attention_histogram": {}, "node_path": ["embed", "lm_head"]}],
    )

    assert trace["schema_version"] == "trace.v1"
    assert trace["run_id"] == "demo-run-id"

