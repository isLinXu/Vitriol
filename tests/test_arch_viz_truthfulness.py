from vitriol.arch_viz.core import Architecture
from vitriol.arch_viz.renderers import html as html_renderer
from vitriol.arch_viz.renderers.html import HTMLRenderer


def test_html_renderer_uses_core_architecture_type() -> None:
    assert html_renderer.Architecture is Architecture
    assert not hasattr(html_renderer, "MODEL_REGISTRY")


def _render_hy3_html(*, total_params: int, parameters: dict) -> str:
    arch = Architecture(
        model_type="hy_v3",
        arch_type="decoder-only",
        total_layers=int(parameters.get("num_layers", 80)),
        total_params=total_params,
        memory_fp16_gb=0.0,
        parameters=parameters,
        features=["Hy3", "MoE", "GQA"],
        layers=[],
    )
    return HTMLRenderer().render_to_string(arch)


def test_hy3_overview_does_not_use_hardcoded_params_or_default_active_params() -> None:
    html = _render_hy3_html(
        total_params=123_000_000_000,
        parameters={
            "dense_prefix_layers": 1,
            "num_experts": 192,
            "top_k_experts": 8,
            "num_shared_experts": 1,
            "mtp_layers": 1,
            "num_kv_heads": 8,
            "num_heads": 64,
            "max_position": 262144,
            # Note: intentionally do not provide active_params_b / activated_params.
        },
    )

    # Truthfulness: must not show hard-coded total params (295B) or default active params (21B).
    assert "295B /" not in html
    assert "/ 21B active" not in html

    # Should display total params derived from arch.total_params (123B).
    assert "123B" in html


def test_hy3_overview_uses_activated_params_when_available() -> None:
    html = _render_hy3_html(
        total_params=123_000_000_000,
        parameters={
            "dense_prefix_layers": 1,
            "num_experts": 192,
            "top_k_experts": 8,
            "num_shared_experts": 1,
            "mtp_layers": 1,
            "num_kv_heads": 8,
            "num_heads": 64,
            "max_position": 262144,
            # DeepSeekAnalyzer and similar tools may populate this field (unit: number of params).
            "activated_params": 21_000_000_000,
        },
    )

    assert "123B / 21B active" in html


def test_html_renderer_escapes_config_derived_strings() -> None:
    malicious = 'bad</title><script>alert("x")</script><span data-x="'
    arch = Architecture(
        model_type=malicious,
        arch_type="decoder-only",
        total_layers=1,
        total_params=1,
        memory_fp16_gb=0.0,
        parameters={"hidden_size": 8, "num_heads": 1},
        features=[malicious],
        layers=[],
    )

    html = HTMLRenderer().render_to_string(arch)

    assert malicious not in html
    assert "&lt;/title&gt;&lt;script&gt;" in html
    assert "\\u003c/script\\u003e" in html
    assert "const ARCH_DATA" in html
