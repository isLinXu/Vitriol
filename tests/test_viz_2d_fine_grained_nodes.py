from pathlib import Path


def test_2d_visualizer_contains_fine_grained_trace_node_ids() -> None:
    """
    The 2D architecture diagram must be able to express fine-grained trace nodes; otherwise we
    cannot achieve 1:1 alignment and explanation.
    This is a static assertion: the template code must include these node-id conventions (as
    data-id/trace_id).
    """
    html = Path("src/vitriol/viz/model_visualizer.html").read_text(encoding="utf-8")

    # attention projections
    assert ":attn:q_proj" in html
    assert ":attn:k_proj" in html
    assert ":attn:v_proj" in html
    assert ":attn:o_proj" in html

    # layer norms
    assert ":norm1" in html
    assert ":norm2" in html

    # ffn projections
    assert ":ffn:gate_proj" in html
    assert ":ffn:up_proj" in html
    assert ":ffn:down_proj" in html


def test_2d_visualizer_prefers_trace_id_for_component_data_id() -> None:
    html = Path("src/vitriol/viz/model_visualizer.html").read_text(encoding="utf-8")
    assert "layer.trace_id" in html
    assert "selectByNodeId" in html
