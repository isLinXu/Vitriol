import copy
import json
import logging
import os
import re
import sys
import threading
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import click

logger = logging.getLogger(__name__)

def build_inline_config_model(model_path):
    """Build a complete model config object from config.json or model_index.json for inline use"""
    config_path = model_path / 'config.json'
    model_index_path = model_path / 'model_index.json'

    if model_index_path.exists():
        # Handle Diffusers format (Stable Diffusion, etc.)
        try:
            model_index = json.loads(model_index_path.read_text(encoding="utf-8"))
            unet_config_path = model_path / 'unet' / 'config.json'
            vae_config_path = model_path / 'vae' / 'config.json'
            text_encoder_config_path = model_path / 'text_encoder' / 'config.json'

            unet_config = json.loads(unet_config_path.read_text(encoding="utf-8")) if unet_config_path.exists() else {}
            vae_config = json.loads(vae_config_path.read_text(encoding="utf-8")) if vae_config_path.exists() else {}
            text_encoder_config = json.loads(text_encoder_config_path.read_text(encoding="utf-8")) if text_encoder_config_path.exists() else {}

            # Extract architecture specifics
            unet_config.get('in_channels', 4)
            unet_layers = len(unet_config.get('down_block_types', [])) * 2 + 1  # rough estimate
            vae_config.get('latent_channels', 4)

            return {
                "name": model_path.name,
                "type": "diffusion",
                "is_diffusion": True,
                "diffusers_type": model_index.get('_class_name', 'StableDiffusionPipeline'),
                "unet_config": unet_config,
                "vae_config": vae_config,
                "text_encoder_config": text_encoder_config,
                "hidden_size": unet_config.get('cross_attention_dim', 768),
                "num_layers": unet_layers,
                "total_params": 0,
                "params_source": "unavailable",
                "config_source": "model_index.json",
                "raw": model_index
            }
        except Exception as e:
            click.echo(f"Warning: Failed to parse diffusers model: {e}", err=True)

    if not config_path.exists():
        return None

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as e:
        click.echo(f"Warning: Failed to read config.json: {e}", err=True)
        return None

    # Try to load meta-config.json (original HF config) first, then fallback to config_meta.json
    meta_config = None
    for meta_name in ('meta-config.json', 'config_meta.json'):
        meta_path = model_path / meta_name
        if meta_path.exists():
            try:
                meta_config = json.loads(meta_path.read_text(encoding="utf-8"))
                break
            except Exception as e:
                click.echo(f"Warning: Failed to read {meta_name}: {e}", err=True)

    # If meta-config.json exists, use it as the authoritative config for visualization
    # while preserving the raw payload for truthfulness/debugging surfaces.
    raw_config = meta_config if meta_config else config
    effective_config = copy.deepcopy(raw_config)

    # Build model config structure matching what JavaScript expects
    model_name = effective_config.get('model_name', model_path.name)

    # If text_config is not present, assume the root config is the text config (standard HF models)
    text_config = effective_config.get('text_config', effective_config)
    vision_config = effective_config.get('vision_config', {})
    audio_config = effective_config.get('audio_config', {})
    tts_config = effective_config.get('tts_config', {})

    # Calculate total params (rough estimate)
    vocab_size = text_config.get('vocab_size', 0)
    hidden_size = text_config.get('hidden_size', 0)
    num_hidden_layers = text_config.get('num_hidden_layers', 0)
    intermediate_size = text_config.get('intermediate_size', 0)
    num_experts = text_config.get('num_experts', 0)

    # Estimate params for dense model
    total_params = vocab_size * hidden_size  # embedding
    total_params += num_hidden_layers * (
        3 * hidden_size * hidden_size +  # QKV projection
        2 * hidden_size * intermediate_size +  # FFN
        4 * hidden_size  # layer norms and residuals
    )
    total_params += vocab_size * hidden_size  # output head

    # Format for JavaScript
    num_layers = num_hidden_layers
    if vision_config:
        num_layers += vision_config.get('num_hidden_layers', 0)
    if audio_config:
        num_layers += audio_config.get('num_hidden_layers', audio_config.get('encoder_layers', 0))
    if tts_config:
        num_layers += tts_config.get('num_hidden_layers', 0)

    # Try to get precise architecture analysis if available
    try:
        from vitriol.arch_viz.analyzer import ArchitectureAnalyzer
        analyzer = ArchitectureAnalyzer()
        # Mock a minimal config object that analyzer can use
        class MockConfig:
            def __init__(self, d):
                self.__dict__.update(d)
                self.model_type = d.get('model_type', 'unknown')

        # Use analyzer for all models to get precise parameter count and architecture details
        arch = analyzer.analyze(MockConfig(effective_config))
        total_params = int(arch.total_params)
        num_layers = arch.total_layers
        hidden_size = arch.parameters.get('hidden_size', hidden_size)
        num_experts = arch.parameters.get('num_experts', num_experts)

        # Update effective_config so that raw JSON has these values correctly for frontend
        effective_config['num_experts'] = num_experts
        effective_config['num_hidden_layers'] = num_layers
        effective_config['hidden_size'] = hidden_size
        effective_config['total_params'] = total_params

        # Add activated_params if available (especially for DeepSeek-V4 MoE)
        activated_params = arch.parameters.get('activated_params', 0)
        if activated_params > 0:
            effective_config['activated_params'] = activated_params

        # Add context length
        max_position = arch.parameters.get('max_position', 0)
        if max_position > 0:
            effective_config['max_position_embeddings'] = max_position

    except Exception as e:
        logger.debug(f"ArchitectureAnalyzer failed for inline config: {e}")

    return {
        "name": model_name,
        "type": effective_config.get('model_type', config.get('model_type', 'unknown')),
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_attention_heads": text_config.get('num_attention_heads', 0),
        "num_key_value_heads": text_config.get('num_key_value_heads', 0),
        "head_dim": hidden_size // text_config.get('num_attention_heads', 1) if text_config.get('num_attention_heads') else 0,
        "intermediate_size": text_config.get('intermediate_size', 0),
        "vocab_size": text_config.get('vocab_size', 0),
        "num_experts": num_experts,
        "total_params": total_params,
        "params_source": "analyzer" if total_params > 0 else "config_derived",
        "activated_params": effective_config.get('activated_params', 0),
        "max_position": effective_config.get('max_position_embeddings', 0),
        "text_config": text_config,  # Include for parseConfig compatibility
        "vision_config": vision_config,
        "audio_config": audio_config,
        "tts_config": tts_config,
        "config_source": (
            "meta-config.json"
            if (model_path / "meta-config.json").exists()
            else "config_meta.json"
            if (model_path / "config_meta.json").exists()
            else "config.json"
        ),
        "raw": raw_config,
        "meta": meta_config
    }


def collect_weight_stats(model_path: Path, max_layers: int = 64) -> dict:
    """Extract weight statistics from real weight shards for the 3D frontend.

    Prefers meta-config.json to recover the original HF architecture config.

    Args:
        model_path: Model directory (weights + meta-config.json).
        max_layers: Max Transformer blocks to sample.

    Returns:
        Frontend-friendly weight stats dictionary.
    """
    try:
        from vitriol.viz.weight_inspector import generate_viz_data
    except ImportError:
        logger.warning("weight_inspector module not available, skipping weight stats")
        return {}

    try:
        viz_data = generate_viz_data(str(model_path), max_layers=max_layers, seed=42)
        # Compact payload: keep only frontend-required fields
        stats = {
            "config_source": viz_data.get("config_source", "unknown"),
            "weight_stats_available": viz_data.get("weight_stats_available", False),
            "total_params": viz_data.get("total_params", 0),
            "model_total_params": viz_data.get("model_total_params", 0),
            "display_params_estimate": viz_data.get("display_params_estimate", 0),
            "params_source": viz_data.get("params_source", "unknown"),
            "sampling": viz_data.get("sampling", {}),
            "layers": [],
        }

        for layer in viz_data.get("layers", []):
            if "sub_layers" in layer:
                # Transformer Block
                block = {
                    "block_index": layer.get("block_index", 0),
                    "sub_layers": [],
                }
                for sl in layer.get("sub_layers", []):
                    sub = {
                        "name": sl.get("name", ""),
                        "type": sl.get("type", ""),
                        "shape": sl.get("shape", []),
                        "params": sl.get("params", 0),
                    }
                    # Keep only key stats
                    if sl.get("stats"):
                        s = sl["stats"]
                        sub["stats"] = {
                            "mean": s.get("mean", 0),
                            "std": s.get("std", 0),
                            "sparsity": s.get("sparsity", 0),
                            "l2_norm": s.get("l2_norm", 0),
                            "is_strided": s.get("is_strided", False),
                            "compression_ratio": s.get("compression_ratio", 1.0),
                        }
                    block["sub_layers"].append(sub)
                stats["layers"].append(block)
            else:
                # Standalone layers (embedding, lm_head)
                entry = {
                    "name": layer.get("name", ""),
                    "type": layer.get("type", ""),
                    "shape": layer.get("shape", []),
                    "params": layer.get("params", 0),
                }
                if layer.get("stats"):
                    s = layer["stats"]
                    entry["stats"] = {
                        "mean": s.get("mean", 0),
                        "std": s.get("std", 0),
                        "sparsity": s.get("sparsity", 0),
                        "l2_norm": s.get("l2_norm", 0),
                        "is_strided": s.get("is_strided", False),
                        "compression_ratio": s.get("compression_ratio", 1.0),
                    }
                stats["layers"].append(entry)

        click.echo(f"  Weight stats: {len(stats['layers'])} layers, "
                    f"available={stats['weight_stats_available']}", err=True)
        return stats

    except Exception as e:
        logger.warning("Failed to collect weight stats: %s", e)
        return {}


class QuietHTTPHandler(SimpleHTTPRequestHandler):
    """HTTP handler that suppresses logging"""
    def log_message(self, format, *args):
        pass  # Suppress request logging


@click.command()
@click.argument('model_path', required=False,
    default=None)
@click.option('--port', '-p', default=8765, help='Port for local server')
@click.option('--no-open', is_flag=True, help='Do not open browser automatically')
@click.option('--3d', 'use_3d', is_flag=True, help='Use 3D visualization (default)')
@click.option('--2d', 'use_2d', is_flag=True, help='Use 2D HTML visualization')
@click.option('--with-weights', 'with_weights', is_flag=True,
              help='Read real weight files and inject stats for visualization')
@click.option(
    '--trace',
    'trace_path',
    type=click.Path(exists=True, dir_okay=False),
    help='Path to trace.json (offline replay injection)',
)
def visualize(model_path, port, no_open, use_3d, use_2d, with_weights, trace_path):
    """Launch interactive visualizer for model architecture.

    MODEL_PATH: Path to ultra model directory or HuggingFace model ID.
                Defaults to the Qwen3.5 MoE demo model.

    Examples:
        vitriol viz                                    # Launch with demo model
        vitriol viz /path/to/ultra/model              # Load custom ultra model
        vitriol viz Qwen/Qwen2-7B                     # Load from HuggingFace
        vitriol viz --2d                              # Use 2D visualization
    """

    # Determine which visualizer to use
    if use_2d:
        visualizer_type = '2d'
    else:
        visualizer_type = '3d'

    # Get the visualizer HTML path
    viz_dir = Path(__file__).parent.parent.parent / 'viz'

    if visualizer_type == '3d':
        html_path = viz_dir / 'model_3d_visualizer.html'
    else:
        html_path = viz_dir / 'model_visualizer.html'

    if not html_path.exists():
        click.echo(f"Error: Visualizer not found at {html_path}", err=True)
        sys.exit(1)

    trace_json = None
    if trace_path:
        try:
            trace_obj = json.loads(Path(trace_path).read_text(encoding="utf-8"))
            trace_json = json.dumps(trace_obj, separators=(",", ":"))
        except Exception as e:
            click.echo(f"Error: Failed to read trace file {trace_path}: {e}", err=True)
            sys.exit(1)

    # Create absolute path for model
    html_file_to_serve = html_path
    inline_config = None

    model_path_for_url = ""
    model_display_name = "Demo"

    if model_path is None:
        click.echo("No model path specified. Using demo data.", err=True)
    else:
        model_path = Path(model_path).resolve()
        model_path_for_url = str(model_path)
        model_display_name = model_path.name

        if model_path.exists():
            config_path = model_path / 'config.json'
            model_index_path = model_path / 'model_index.json'

            if config_path.exists() or model_index_path.exists():
                click.echo(f"Info: Loading model config from {model_path}", err=True)

                # Build inline config
                inline_config = build_inline_config_model(model_path)

                # Optionally collect weight statistics from real weight files
                weight_stats = {}
                if with_weights:
                    click.echo(f"Info: Reading weight files from {model_path}...", err=True)
                    weight_stats = collect_weight_stats(model_path)
                # Auto-detect logic removed to allow viewing raw architecture of ultra models without weight interference

                # Create a temporary customized HTML with model path
                import tempfile
                temp_dir = Path(tempfile.mkdtemp(prefix="vitriol_viz_"))
                temp_html = temp_dir / f'model_visualizer_{port}.html'

                html_content = html_path.read_text(encoding="utf-8")

                # Replace the built-in demo model path with the requested model.
                html_content = re.sub(
                    r"modelPath\s*=\s*urlParams\.get\('model'\)\s*\|\|\s*'[^']*';",
                    f"modelPath = urlParams.get('model') || '{model_path}';",
                    html_content,
                    count=1,
                )
                html_content = re.sub(
                    r"modelPath\s*=\s*'output/[^']*';",
                    f"modelPath = '{model_path}';",
                    html_content,
                    count=1,
                )

                # Inject inline config to bypass fetch (for file:// protocol)
                if inline_config:
                    inline_config_json = json.dumps(inline_config, separators=(',', ':'))
                    html_content = html_content.replace(
                        '// INLINE_CONFIG_MARKER',
                        f'window.__INLINE_MODEL_CONFIG__ = {inline_config_json};'
                    )

                # Inject ArchitectureAnalyzer layer details for 2D truthfulness (optional).
                # This allows the 2D visualizer to reuse backend-derived layer breakdown instead of JS heuristics.
                inline_arch_data = None
                if inline_config and isinstance(inline_config, dict) and inline_config.get("raw"):
                    try:
                        from vitriol.arch_viz.analyzer import ArchitectureAnalyzer

                        analyzer = ArchitectureAnalyzer()

                        class MockConfig:
                            def __init__(self, d):
                                self.__dict__.update(d)
                                self.model_type = d.get("model_type", "unknown")

                        arch = analyzer.analyze(MockConfig(inline_config["raw"]))

                        layers_payload = []
                        for layer in getattr(arch, "layers", []) or []:
                            layers_payload.append(
                                {
                                    "name": getattr(layer, "name", ""),
                                    "type": getattr(layer, "type", ""),
                                    "params": int(getattr(layer, "params", 0) or 0),
                                    "shape": list(getattr(layer, "shape", ()) or ()),
                                    "description": getattr(layer, "description", "") or "",
                                }
                            )

                        inline_arch_data = {
                            "model_type": getattr(arch, "model_type", "") or "unknown",
                            "arch_type": getattr(arch, "arch_type", "") or "",
                            "total_layers": int(getattr(arch, "total_layers", 0) or 0),
                            "total_params": int(getattr(arch, "total_params", 0) or 0),
                            "memory_fp16_gb": float(getattr(arch, "memory_fp16_gb", 0.0) or 0.0),
                            "parameters": getattr(arch, "parameters", {}) or {},
                            "features": getattr(arch, "features", []) or [],
                            "layers": layers_payload,
                            "source": "ArchitectureAnalyzer",
                        }
                    except Exception as e:
                        logger.debug("ArchitectureAnalyzer failed for INLINE_ARCH_MARKER injection: %s", e)

                if inline_arch_data:
                    inline_arch_json = json.dumps(inline_arch_data, separators=(",", ":"))
                    html_content = html_content.replace(
                        "// INLINE_ARCH_MARKER",
                        f"window.__INLINE_ARCH_DATA__ = {inline_arch_json};",
                    )

                # Inject weight statistics for 3D visualization
                if weight_stats:
                    weight_stats_json = json.dumps(weight_stats, separators=(',', ':'))
                    html_content = html_content.replace(
                        '// INLINE_WEIGHT_STATS_MARKER',
                        f'window.__INLINE_WEIGHT_STATS__ = {weight_stats_json};'
                    )

                # Inject offline trace for replay (optional).
                if trace_json:
                    html_content = html_content.replace(
                        "// INLINE_TRACE_MARKER",
                        f"window.__VITRIOL_TRACE__ = {trace_json};",
                    )

                temp_html.write_text(html_content, encoding="utf-8")
                html_file_to_serve = temp_html
            else:
                click.echo(f"Warning: No config.json found in {model_path}, using demo data", err=True)
                html_file_to_serve = html_path
        else:
            click.echo("Info: Model path not found, using demo data", err=True)
            html_file_to_serve = html_path

    # If we're serving the original HTML (demo mode) but still want trace injection,
    # create a temporary customized HTML to avoid mutating the source file.
    if trace_json and html_file_to_serve == html_path:
        import tempfile

        temp_dir = Path(tempfile.mkdtemp(prefix="vitriol_viz_trace_"))
        temp_html = temp_dir / f"model_visualizer_trace_{port}.html"
        html_content = html_path.read_text(encoding="utf-8")
        html_content = html_content.replace(
            "// INLINE_TRACE_MARKER",
            f"window.__VITRIOL_TRACE__ = {trace_json};",
        )
        temp_html.write_text(html_content, encoding="utf-8")
        html_file_to_serve = temp_html

    # Start HTTP server
    os.chdir(html_file_to_serve.parent)

    # Handler that serves the specific HTML file
    class CustomHandler(QuietHTTPHandler):
        def do_GET(self):
            if self.path == '/' or self.path == '':
                self.path = '/' + html_file_to_serve.name
            return SimpleHTTPRequestHandler.do_GET(self)

    # Find an available port if the default is in use
    max_retries = 50
    for _ in range(max_retries):
        try:
            httpd = HTTPServer(('127.0.0.1', port), CustomHandler)
            break
        except OSError as e:
            if getattr(e, 'errno', 0) in (48, 98, 10048) or "in use" in str(e).lower():
                click.echo(f"Port {port} is in use, trying {port + 1}...")
                port += 1
            else:
                click.echo(f"Error binding to port {port}: {e}")
                port += 1
    else:
        try:
            click.echo(f"Could not find an available port in {max_retries} attempts. Asking OS to assign one...")
            httpd = HTTPServer(('127.0.0.1', 0), CustomHandler)
            port = httpd.server_port
            click.echo(f"OS assigned port {port}.")
        except OSError as e:
            click.echo(f"Error: Could not start server. {e}", err=True)
            sys.exit(1)

    # Start server in background thread
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()

    url = f'http://localhost:{port}/{html_file_to_serve.name}#?model={model_path_for_url}'

    # Open browser
    if not no_open:
        def open_browser():
            import time
            time.sleep(1)
            webbrowser.open(url)

        thread = threading.Thread(target=open_browser)
        thread.daemon = True
        thread.start()

    click.echo(f"\n{'='*60}")
    click.echo("  Vitriol Model Visualizer")
    click.echo(f"{'='*60}")
    click.echo(f"  Model: {model_display_name}")
    click.echo(f"  Type:  {visualizer_type.upper()} Interactive")
    click.echo(f"  URL:   {url}")
    click.echo(f"  Server: http://localhost:{port}")
    click.echo(f"{'='*60}")
    click.echo("\n  Controls:")
    click.echo("    - Drag: Rotate view")
    click.echo("    - Scroll: Zoom in/out")
    click.echo("    - Click: Select layer for details")
    click.echo("    - R: Reset camera position")
    click.echo("\n  Press Ctrl+C to exit\n")

    # Keep running
    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        click.echo("\n\nExiting...")
        httpd.shutdown()
        sys.exit(0)
