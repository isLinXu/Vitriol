from __future__ import annotations


def _assert_trace_v1_min_fields(trace: dict) -> None:
    assert trace["schema_version"] == "trace.v1"

    # tokens (minimum fields)
    tokens = trace["tokens"]
    assert isinstance(tokens, dict)
    assert isinstance(tokens["prompt_token_ids"], list) and all(isinstance(x, int) for x in tokens["prompt_token_ids"])
    assert isinstance(tokens["prompt_tokens"], list) and all(isinstance(x, str) for x in tokens["prompt_tokens"])
    assert isinstance(tokens["generated_token_ids"], list) and all(
        isinstance(x, int) for x in tokens["generated_token_ids"]
    )
    assert isinstance(tokens["generated_tokens"], list) and all(isinstance(x, str) for x in tokens["generated_tokens"])

    # events (minimum fields)
    events = trace["events"]
    assert isinstance(events, list) and events

    e0 = events[0]
    assert isinstance(e0, dict)
    assert "token_index" in e0 and isinstance(e0["token_index"], int)
    assert "phase" in e0 and isinstance(e0["phase"], str)
    # Token-level viz: an event should carry token text and a global token index (prompt+generated merged view).
    assert "token_text" in e0 and isinstance(e0["token_text"], str)
    assert "token_global_index" in e0 and isinstance(e0["token_global_index"], int)
    # Attention view: an event should optionally carry attention_topk (key=block:{i}:attn).
    assert "attention_topk" in e0 and isinstance(e0["attention_topk"], dict)
    # Attention distribution (bucketized): used for the "full distribution heat bar".
    assert "attention_histogram" in e0 and isinstance(e0["attention_histogram"], dict)
    # Must be emitted per-layer (key=block:{i}:attn) to avoid semantic mismatch between display and highlighting.
    assert "block:0:attn" in e0["attention_histogram"]
    h0 = e0["attention_histogram"]["block:0:attn"]
    assert isinstance(h0, dict)
    assert "bins" in h0 and isinstance(h0["bins"], int)
    assert "values" in h0 and isinstance(h0["values"], list)

    # node_path（embed -> ... -> lm_head）
    node_path = e0["node_path"]
    assert isinstance(node_path, list) and all(isinstance(x, str) for x in node_path)
    assert node_path[0] == "embed"
    assert node_path[-1] == "lm_head"


def test_trace_schema_v1_min_fields() -> None:
    # Validate the schema structure only (does not depend on actual model inference).
    trace = {
        "schema_version": "trace.v1",
        "model_path": "output/tinyllama-hybrid-ultra-test",
        "prompt": "hello",
        "max_new_tokens": 8,
        "tokens": {
            "prompt_token_ids": [1],
            "prompt_tokens": ["hello"],
            "generated_token_ids": [2],
            "generated_tokens": ["!"],
        },
        "events": [
            {
                "token_index": 0,
                "token_global_index": 0,
                "token_text": "hello",
                "phase": "prefill",
                "attention_topk": {"block:0:attn": [{"src": 0, "w": 1.0}]},
                "attention_histogram": {
                    "block:0:attn": {"bins": 8, "values": [1, 0, 0, 0, 0, 0, 0, 0]},
                    "block:1:attn": {"bins": 8, "values": [0, 1, 0, 0, 0, 0, 0, 0]},
                },
                "node_path": ["embed", "block:0:attn", "block:0:mlp", "lm_head"],
            }
        ],
    }

    _assert_trace_v1_min_fields(trace)
