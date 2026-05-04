"""
Vitriol Web UI - Gradio Application
==================================

A web-based user interface for Vitriol features.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, Optional


def _ensure_cache_dirs() -> None:
    """Point third-party caches at writable temp directories."""
    base = Path(tempfile.gettempdir()) / "vitriol-runtime-cache"
    mpl_dir = base / "matplotlib"
    xdg_dir = base / "xdg"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    xdg_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_dir))


_ensure_cache_dirs()

import gradio as gr  # noqa: E402

from vitriol.evolution import (  # noqa: E402
    EvolutionTree,
    TreeVisualizer,
    ArchComparator,
    ComparisonReport,
    ArchSimulator,
    InnovationTimeline,
    ArchitectureRecommender,
    UseCase,
)
from vitriol.nas.targeted_nas import (  # noqa: E402
    ConstraintOptimizer,
    ConstraintType,
    Constraint,
    OptimizationTarget,
    ObjectiveType,
)
from vitriol.nas.search_space import LLMSearchSpace  # noqa: E402

logger = logging.getLogger(__name__)


def load_model_config(model_id: str, trust_remote_code: bool = True) -> Optional[Dict]:
    """Load model configuration from HuggingFace."""
    try:
        from vitriol.utils.hf_loading import load_config as hf_load_config

        config = hf_load_config(
            model_id,
            security={
                "trust_remote_code": trust_remote_code,
                "allow_network": True,
                "local_files_only": False,
            },
        )
        return config.to_dict()
    except Exception as e:
        logger.warning(f"Failed to load config for {model_id}: {e}")
        return None


def format_params(params: int) -> str:
    """Format parameter count for display."""
    if params >= 1e12:
        return f"{params / 1e12:.1f}T"
    elif params >= 1e9:
        return f"{params / 1e9:.1f}B"
    elif params >= 1e6:
        return f"{params / 1e6:.1f}M"
    else:
        return f"{params:,}"


def create_app(
    title: str = "Vitriol - LLM Architecture Explorer",
) -> gr.Blocks:
    """
    Create the Vitriol Gradio application.

    Args:
        title: Application title

    Returns:
        Gradio Blocks application
    """

    app = gr.Blocks(
        title=title,
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="purple"),
        head='<style>\
            .vitriol-header { text-align: center; padding: 20px; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 20px; }\
            .metric-card { background: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #667eea; }\
            .innovation-tag { display: inline-block; background: #e8f4fd; padding: 4px 8px; border-radius: 4px; margin: 2px; font-size: 12px; }\
        </style>',
    )

    with app:
        # Header
        gr.HTML(f'''
            <div class="vitriol-header">
                <h1>🏛️ {title}</h1>
                <p>Explore, Visualize, and Optimize LLM Architectures</p>
            </div>
        ''')

        # Main tabs
        with gr.Tabs():
            # ─────────────────────────────────────────────────────────────────
            # Tab 1: Model Comparison
            # ─────────────────────────────────────────────────────────────────
            with gr.TabItem("⚖️ Model Comparison"):
                gr.Markdown("""
                ## Compare Two Model Architectures

                Enter two HuggingFace model IDs to get a detailed architecture comparison report.
                """)

                with gr.Row():
                    with gr.Column():
                        model1_id = gr.Textbox(
                            label="Model 1",
                            placeholder="e.g., Qwen/Qwen2.5-7B",
                            value="Qwen/Qwen2.5-7B",
                        )
                        model2_id = gr.Textbox(
                            label="Model 2",
                            placeholder="e.g., meta-llama/Llama-3-8B",
                            value="meta-llama/Llama-3-8B",
                        )
                        compare_trc = gr.Checkbox(
                            label="Trust Remote Code (⚠️ may execute remote code)",
                            value=True,
                        )
                        compare_btn = gr.Button("🔄 Compare", variant="primary")

                    with gr.Column():
                        comparison_output = gr.Markdown(
                            label="Comparison Report",
                        )

                gr.Examples(
                    examples=[
                        ["Qwen/Qwen2.5-7B", "meta-llama/Llama-3-8B"],
                        ["deepseek-ai/DeepSeek-V2", "Qwen/Qwen2.5-72B"],
                    ],
                    inputs=[model1_id, model2_id],
                )

                def compare_models(m1_id, m2_id, trc, progress=gr.Progress()):
                    """Compare two model architectures."""
                    try:
                        progress(0, desc="Loading model 1...")
                        config1 = load_model_config(m1_id, trust_remote_code=trc)
                        progress(0.33, desc="Loading model 2...")
                        config2 = load_model_config(m2_id, trust_remote_code=trc)
                        progress(0.66, desc="Comparing...")

                        if not config1 or not config2:
                            return "❌ Failed to load one or both model configs"

                        comparator = ArchComparator()
                        result = comparator.compare_from_ids(m1_id, m2_id, config1, config2)
                        report = ComparisonReport.to_markdown(result)

                        progress(1.0, desc="Complete!")
                        return report
                    except Exception as e:
                        return f"❌ Error: {str(e)}"

                compare_btn.click(
                    compare_models,
                    inputs=[model1_id, model2_id, compare_trc],
                    outputs=[comparison_output],
                )

            # ─────────────────────────────────────────────────────────────────
            # Tab 2: Evolution Tree
            # ─────────────────────────────────────────────────────────────────
            with gr.TabItem("🌳 Evolution Tree"):
                gr.Markdown("""
                ## Architecture Evolution Tree

                Visualize the family tree of LLM architectures and their innovations.
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        tree_title = gr.Textbox(
                            label="Tree Title",
                            value="LLM Architecture Evolution",
                        )
                        show_innovations = gr.Checkbox(
                            label="Show Innovations",
                            value=True,
                        )
                        build_tree_btn = gr.Button("🌲 Build Tree", variant="primary")

                    with gr.Column(scale=2):
                        tree_output = gr.HTML(
                            label="Evolution Tree Visualization",
                        )

                with gr.Row():
                    family_list = gr.JSON(label="Family Data")

                def build_tree(title, show_inn):
                    """Build and visualize the architecture evolution tree."""
                    try:
                        tree = EvolutionTree()
                        tree.build()

                        visualizer = TreeVisualizer(tree)
                        html = visualizer.generate_html_string(
                            title=title,
                            show_innovations=show_inn,
                        )

                        # Get family data
                        family_data = {
                            family_name: {
                                "root": data.get("root", "Unknown"),
                                "members": len(data.get("members", {})),
                            }
                            for family_name, data in tree.families.items()
                        }

                        return html, family_data
                    except Exception as e:
                        return f"<h3>❌ Error: {str(e)}</h3>", {}

                build_tree_btn.click(
                    build_tree,
                    inputs=[tree_title, show_innovations],
                    outputs=[tree_output, family_list],
                )

            # ─────────────────────────────────────────────────────────────────
            # Tab 3: Targeted NAS
            # ─────────────────────────────────────────────────────────────────
            with gr.TabItem("🎯 Targeted NAS"):
                gr.Markdown("""
                ## Targeted Neural Architecture Search

                Find the optimal LLM architecture under your constraints.
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        target_vram = gr.Slider(
                            label="Max VRAM (GB)",
                            minimum=1,
                            maximum=128,
                            value=24,
                            step=1,
                        )
                        target_params = gr.Slider(
                            label="Max Parameters (B)",
                            minimum=1,
                            maximum=200,
                            value=70,
                            step=1,
                        )
                        objective = gr.Dropdown(
                            label="Optimization Objective",
                            choices=[
                                "Maximize Efficiency (Params/VRAM)",
                                "Minimize Parameters",
                                "Minimize VRAM",
                            ],
                            value="Maximize Efficiency (Params/VRAM)",
                        )
                        nas_iterations = gr.Slider(
                            label="Search Iterations",
                            minimum=10,
                            maximum=200,
                            value=50,
                            step=10,
                        )
                        run_nas_btn = gr.Button("🚀 Run Targeted NAS", variant="primary")

                    with gr.Column(scale=2):
                        nas_results = gr.JSON(label="Search Results")
                        nas_summary = gr.Markdown()

                def run_targeted_nas(vram_limit, param_limit, objective_choice, iterations, progress=gr.Progress()):
                    """Run targeted NAS search."""
                    try:
                        progress(0, desc="Initializing optimizer...")

                        optimizer = ConstraintOptimizer()

                        # Add constraints
                        if vram_limit:
                            optimizer.add_constraint(Constraint(ConstraintType.MAX_VRAM, float(vram_limit)))
                        if param_limit:
                            optimizer.add_constraint(Constraint(ConstraintType.MAX_PARAMS, float(param_limit) * 1e6))

                        # Set objective
                        if "Maximize" in objective_choice:
                            optimizer.add_objective(OptimizationTarget(ObjectiveType.MAXIMIZE_EFFICIENCY))
                        elif "Minimize Parameters" in objective_choice:
                            optimizer.add_objective(OptimizationTarget(ObjectiveType.MINIMIZE_PARAMS))
                        else:
                            optimizer.add_objective(OptimizationTarget(ObjectiveType.MINIMIZE_VRAM))

                        progress(0.1, desc=f"Running {iterations} iterations...")

                        gene, score, metrics = optimizer.optimize(
                            LLMSearchSpace(),
                            None,
                            n_iterations=int(iterations),
                            verbose=False,
                        )

                        progress(1.0, desc="Complete!")

                        result_dict = {
                            "architecture": gene.to_dict(),
                            "metrics": metrics,
                            "optimization_score": score,
                        }

                        summary = f"""
## 🎯 Targeted NAS Results

### Best Architecture Found
| Parameter | Value |
|-----------|-------|
| Layers | {gene.n_layers} |
| Hidden Size | {gene.hidden_size} |
| Attention | {gene.attention_type} |
| FFN | {gene.ffn_type} |
| Vocab Size | {gene.vocab_size:,} |

### Performance Metrics
| Metric | Value |
|--------|-------|
| Total Params | {metrics.get('params_millions', 0):.1f}M |
| Active Params | {metrics.get('active_params_millions', 0):.1f}M |
| VRAM (Full) | {metrics.get('vram_gb', 0):.2f} GB |
| VRAM (Inference) | {metrics.get('vram_inference_gb', 0):.2f} GB |
| FLOPs/param | {metrics.get('flops_per_param', 0):.1f} |

### Optimization
- Objective: {objective_choice}
- Iterations: {iterations}
- Efficiency Score: {score:.2f}
"""
                        return result_dict, summary
                    except Exception as e:
                        return {}, f"❌ Error: {str(e)}"

                run_nas_btn.click(
                    run_targeted_nas,
                    inputs=[target_vram, target_params, objective, nas_iterations],
                    outputs=[nas_results, nas_summary],
                )

            # ─────────────────────────────────────────────────────────────────
            # Tab 4: Architecture Simulator
            # ─────────────────────────────────────────────────────────────────
            with gr.TabItem("⚡ Architecture Simulator"):
                gr.Markdown("""
                ## Architecture Performance Simulator

                Estimate VRAM usage, FLOPs, and latency for a model architecture.
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        sim_model_id = gr.Textbox(
                            label="Model ID",
                            placeholder="e.g., Qwen/Qwen2.5-72B",
                            value="Qwen/Qwen2.5-7B",
                        )
                        sim_batch_size = gr.Slider(
                            label="Batch Size",
                            minimum=1,
                            maximum=256,
                            value=1,
                            step=1,
                        )
                        sim_seq_length = gr.Slider(
                            label="Sequence Length",
                            minimum=1,
                            maximum=32768,
                            value=512,
                            step=1,
                        )
                        sim_dtype = gr.Dropdown(
                            label="Data Type",
                            choices=["bfloat16", "float16", "float32"],
                            value="bfloat16",
                        )
                        sim_gpu = gr.Dropdown(
                            label="GPU",
                            choices=["H100", "A100", "A6000", "RTX 4090", "V100"],
                            value="A100",
                        )
                        sim_trc = gr.Checkbox(
                            label="Trust Remote Code (⚠️ may execute remote code)",
                            value=True,
                        )
                        simulate_btn = gr.Button("⚡ Simulate", variant="primary")

                    with gr.Column(scale=2):
                        sim_results = gr.JSON(label="Simulation Results")
                        sim_summary = gr.Markdown()

                def simulate_architecture(model_id, batch_size, seq_length, dtype, gpu, trc, progress=gr.Progress()):
                    """Simulate architecture performance."""
                    try:
                        progress(0, desc="Loading configuration...")

                        config = load_model_config(model_id, trust_remote_code=trc)
                        if not config:
                            return {}, "❌ Failed to load configuration"

                        progress(0.5, desc="Running simulation...")

                        simulator = ArchSimulator(dtype=dtype, gpu_model=gpu)
                        result = simulator.simulate(model_id, config, batch_size, seq_length)

                        progress(1.0, desc="Complete!")

                        result_dict = result.to_dict()

                        summary = f"""
## ⚡ Simulation Results: {model_id}

### Configuration
| Setting | Value |
|---------|-------|
| Batch Size | {batch_size} |
| Sequence Length | {seq_length} |
| Data Type | {dtype} |
| GPU | {gpu} |

### Parameters
| Type | Count |
|------|-------|
| Total | {format_params(result_dict.get('total_params', 0))} |
| Active per token | {format_params(result_dict.get('active_params_per_token', 0))} |

### Memory (VRAM)
| Mode | Usage |
|------|-------|
| Full Model | {result_dict.get('vram_full_model', 0):.2f} GB |
| Inference | {result_dict.get('vram_inference', 0):.2f} GB |
| Training | {result_dict.get('vram_training', 0):.2f} GB |

### Performance
| Metric | Value |
|--------|-------|
| FLOPs/token | {result_dict.get('flops_per_token', 0):,.0f} |
| Tokens/sec | {result_dict.get('tokens_per_second', 0):.1f} |
| Latency | {result_dict.get('inference_latency_ms', 0):.2f} ms/token |

### Efficiency
- **Params/GB VRAM**: {result_dict.get('params_per_vram', 0):.2f}M
"""
                        return result_dict, summary
                    except Exception as e:
                        return {}, f"❌ Error: {str(e)}"

                simulate_btn.click(
                    simulate_architecture,
                    inputs=[sim_model_id, sim_batch_size, sim_seq_length, sim_dtype, sim_gpu, sim_trc],
                    outputs=[sim_results, sim_summary],
                )

            # ─────────────────────────────────────────────────────────────────
            # Tab 5: Architecture Scorecard
            # ─────────────────────────────────────────────────────────────────
            with gr.TabItem("📋 Architecture Scorecard"):
                gr.Markdown("""
                ## Architecture Scorecard

                Get a standardized architecture score for any LLM model.
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        score_model_id = gr.Textbox(
                            label="Model ID",
                            placeholder="e.g., Qwen/Qwen2.5-72B",
                            value="Qwen/Qwen2.5-7B",
                        )
                        score_trc = gr.Checkbox(
                            label="Trust Remote Code (⚠️ may execute remote code)",
                            value=True,
                        )
                        generate_score_btn = gr.Button("📊 Generate Scorecard", variant="primary")

                    with gr.Column(scale=2):
                        score_output = gr.HTML(label="Scorecard")

                def generate_scorecard(model_id, trc, progress=gr.Progress()):
                    """Generate architecture scorecard."""
                    try:
                        progress(0, desc="Loading configuration...")
                        config = load_model_config(model_id, trust_remote_code=trc)
                        if not config:
                            return "<h3>❌ Failed to load model config</h3>"

                        progress(0.5, desc="Analyzing architecture...")

                        # Extract key metrics
                        hidden_size = config.get('hidden_size', 0)
                        num_layers = config.get('num_hidden_layers', 0)
                        num_heads = config.get('num_attention_heads', 0)
                        num_kv_heads = config.get('num_key_value_heads', num_heads)
                        vocab_size = config.get('vocab_size', 0)
                        intermediate_size = config.get('intermediate_size', 0)
                        max_pos = config.get('max_position_embeddings', 0)

                        # Calculate basic params
                        params = 0
                        if hidden_size and num_layers and vocab_size:
                            # Rough estimate
                            params = vocab_size * hidden_size  # embeddings
                            params += 3 * hidden_size * hidden_size * num_layers  # qkv
                            params += 2 * num_layers * hidden_size * intermediate_size  # ffn

                        # Calculate scores
                        gqa_ratio = num_kv_heads / num_heads if num_heads > 0 else 1
                        context_score = min(1.0, max_pos / 32768)
                        efficiency_score = gqa_ratio * 0.5 + (1 - gqa_ratio) * 0.5

                        # Estimate overall score
                        overall = int(60 + efficiency_score * 20 + context_score * 20)

                        progress(1.0, desc="Complete!")

                        return f"""
                        <div class="metric-card" style="max-width: 600px;">
                            <h2 style="color: #667eea;">📋 Scorecard: {model_id}</h2>
                            <hr style="margin: 15px 0; border-color: #eee;">

                            <div style="display: flex; justify-content: space-around; text-align: center;">
                                <div>
                                    <div style="font-size: 48px; color: #667eea;">{overall}</div>
                                    <div style="color: #888;">Overall Score</div>
                                </div>
                            </div>

                            <h3 style="margin-top: 20px;">Key Parameters</h3>
                            <table style="width: 100%; margin-top: 10px;">
                                <tr><td><b>Hidden Size</b></td><td>{hidden_size:,}</td></tr>
                                <tr><td><b>Layers</b></td><td>{num_layers}</td></tr>
                                <tr><td><b>Attention Heads</b></td><td>{num_heads}</td></tr>
                                <tr><td><b>KV Heads</b></td><td>{num_kv_heads}</td></tr>
                                <tr><td><b>GQA Ratio</b></td><td>{gqa_ratio:.2f}</td></tr>
                                <tr><td><b>Vocab Size</b></td><td>{vocab_size:,}</td></tr>
                                <tr><td><b>Max Position</b></td><td>{max_pos:,}</td></tr>
                                <tr><td><b>Est. Params</b></td><td>{format_params(params)}</td></tr>
                            </table>

                            <h3 style="margin-top: 20px;">Scores</h3>
                            <table style="width: 100%; margin-top: 10px;">
                                <tr>
                                    <td>Parameter Efficiency</td>
                                    <td>
                                        <div style="background: #eee; border-radius: 4px; height: 20px; width: 100%;">
                                            <div style="background: #10b981; border-radius: 4px; height: 20px; width: {efficiency_score*100:.0f}%;"></div>
                                        </div>
                                    </td>
                                    <td>{efficiency_score*100:.0f}%</td>
                                </tr>
                                <tr>
                                    <td>Context Support</td>
                                    <td>
                                        <div style="background: #eee; border-radius: 4px; height: 20px; width: 100%;">
                                            <div style="background: #f59e0b; border-radius: 4px; height: 20px; width: {context_score*100:.0f}%;"></div>
                                        </div>
                                    </td>
                                    <td>{context_score*100:.0f}%</td>
                                </tr>
                            </table>
                        </div>
                        """
                    except Exception as e:
                        return f"<h3>❌ Error: {str(e)}</h3>"

                generate_score_btn.click(
                    generate_scorecard,
                    inputs=[score_model_id, score_trc],
                    outputs=[score_output],
                )

            # ─────────────────────────────────────────────────────────────────
            # Tab 6: Innovation Timeline
            # ─────────────────────────────────────────────────────────────────
            with gr.TabItem("📅 Innovation Timeline"):
                gr.Markdown("""
                ## LLM Architecture Innovation Timeline

                Explore the chronological evolution of LLM architectures and their key innovations.
                """)

                with gr.Row():
                    timeline_title = gr.Textbox(
                        label="Timeline Title",
                        value="LLM Architecture Innovation Timeline",
                    )
                    show_timeline_btn = gr.Button("📅 Generate Timeline", variant="primary")

                timeline_output = gr.HTML(label="Timeline Visualization")

                def generate_timeline(title):
                    """Generate the innovation timeline."""
                    try:
                        timeline = InnovationTimeline()
                        timeline.build_events()
                        html = timeline.generate_html(title=title)
                        return html
                    except Exception as e:
                        return f"<h3>❌ Error: {str(e)}</h3>"

                show_timeline_btn.click(
                    generate_timeline,
                    inputs=[timeline_title],
                    outputs=[timeline_output],
                )

            # ─────────────────────────────────────────────────────────────────
            # Tab 7: Architecture Recommender
            # ─────────────────────────────────────────────────────────────────
            with gr.TabItem("🎯 Architecture Recommender"):
                gr.Markdown("""
                ## Architecture Recommender

                Find the best LLM architecture for your requirements.
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Requirements")

                        req_max_params = gr.Slider(
                            label="Max Parameters (B)",
                            minimum=0.5,
                            maximum=200,
                            value=7,
                            step=0.5,
                        )
                        req_max_vram = gr.Slider(
                            label="Max VRAM (GB)",
                            minimum=1,
                            maximum=128,
                            value=24,
                            step=1,
                        )
                        req_use_case = gr.Dropdown(
                            label="Primary Use Case",
                            choices=["general", "chat", "code", "embedding", "long_context"],
                            value="general",
                        )
                        req_prefer_moe = gr.Checkbox(
                            label="Prefer MoE Architectures",
                            value=False,
                        )
                        req_require_gqa = gr.Checkbox(
                            label="Require GQA Support",
                            value=False,
                        )
                        req_long_context = gr.Checkbox(
                            label="Require 128K+ Context",
                            value=False,
                        )
                        recommend_btn = gr.Button("🎯 Get Recommendations", variant="primary")

                    with gr.Column(scale=2):
                        recommend_output = gr.JSON(label="Recommendations")
                        recommend_summary = gr.Markdown()

                def get_recommendations(max_params, max_vram, use_case, prefer_moe, require_gqa, long_context, progress=gr.Progress()):
                    """Get architecture recommendations."""
                    try:
                        progress(0, desc="Initializing recommender...")

                        recommender = ArchitectureRecommender()

                        progress(0.3, desc="Finding best architectures...")

                        recs = recommender.recommend(
                            max_params=max_params,
                            max_vram=max_vram,
                            use_case=UseCase(use_case),
                            prefer_moe=prefer_moe,
                            require_gqa=require_gqa,
                            require_long_context=long_context,
                        )

                        progress(1.0, desc="Complete!")

                        if not recs:
                            return {}, "No matching architectures found. Try relaxing constraints."

                        # Format results
                        rec_list = []
                        for rec in recs[:5]:
                            rec_list.append({
                                "model_id": rec.model_id,
                                "family": rec.family,
                                "params_b": round(rec.params_b, 1),
                                "vram_gb": round(rec.vram_gb, 1),
                                "score": round(rec.score, 2),
                                "reasons": rec.match_reasons[:3],
                                "innovations": rec.innovations[:5],
                            })

                        summary = f"## 🎯 Found {len(rec_list)} Recommendations\n\n"
                        for i, rec in enumerate(rec_list, 1):
                            summary += f"### {i}. {rec['model_id']}\n"
                            summary += f"- **Family:** {rec['family']}\n"
                            summary += f"- **Parameters:** {rec['params_b']}B\n"
                            summary += f"- **VRAM:** {rec['vram_gb']}GB\n"
                            summary += f"- **Score:** {rec['score']}\n"
                            if rec['reasons']:
                                summary += f"- **Why:** {', '.join(rec['reasons'])}\n"
                            summary += "\n"

                        return {"recommendations": rec_list, "total": len(rec_list)}, summary

                    except Exception as e:
                        return {}, f"❌ Error: {str(e)}"

                recommend_btn.click(
                    get_recommendations,
                    inputs=[req_max_params, req_max_vram, req_use_case, req_prefer_moe, req_require_gqa, req_long_context],
                    outputs=[recommend_output, recommend_summary],
                )

        # Footer
        gr.HTML('''
            <div style="text-align: center; padding: 20px; color: #666;">
                <p>🏛️ Vitriol - LLM Architecture Explorer</p>
                <p><a href="https://github.com/isLinXu/Vitriol">GitHub</a> | <a href="https://vitriol.readthedocs.io">Documentation</a></p>
            </div>
        ''')

    return app


def launch(
    share: bool = False,
    port: Optional[int] = None,
    debug: bool = False,
) -> None:
    """
    Launch the Vitriol web UI.

    Args:
        share: Whether to create a public share link
        port: Port to run on (default: 7860)
        debug: Enable debug mode
    """
    app = create_app()
    app.launch(
        share=share,
        server_port=port or 7860,
        debug=debug,
    )


if __name__ == "__main__":
    launch()
