"""Weight visualization command — reads real weight files (safetensors/.bin)
and extracts per-layer statistics for 3D visualization.

Uses meta-config.json to recover original model architecture parameters
when the config.json has been shrunk by Ultra/HybridUltra strategies.
"""

import click
import logging
import json
import os
import tempfile
import threading
import time
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class QuietHTTPHandler(SimpleHTTPRequestHandler):
    """HTTP handler that suppresses logging"""
    def log_message(self, format, *args):
        pass

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

    def guess_type(self, path):
        """Force correct content type for JSON"""
        if path.endswith('.json'):
            return 'application/json'
        return super().guess_type(path)


def serve_3d_weights(temp_dir: str, port: int = 8781, no_open: bool = False):
    max_retries = 50
    for _ in range(max_retries):
        try:
            httpd = HTTPServer(
                ('127.0.0.1', port),
                lambda *args, **kwargs: QuietHTTPHandler(*args, directory=temp_dir, **kwargs),
            )
            break
        except OSError as e:
            if getattr(e, 'errno', 0) in (48, 98, 10048) or "in use" in str(e).lower():
                port += 1
            else:
                port += 1
    else:
        try:
            httpd = HTTPServer(
                ('127.0.0.1', 0),
                lambda *args, **kwargs: QuietHTTPHandler(*args, directory=temp_dir, **kwargs),
            )
            port = httpd.server_port
        except OSError as e:
            click.echo(f"Error: Could not start server. {e}", err=True)
            return

    url = f"http://127.0.0.1:{port}/weight_3d_visualizer.html"
    click.echo(f"\nStarting 3D Weight Matrix Visualizer at: {url}")
    click.echo("Press Ctrl+C to stop the server.")

    if not no_open:
        threading.Thread(
            target=lambda: (time.sleep(1), webbrowser.open(url)), daemon=True
        ).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        click.echo("\nShutting down server...")
    finally:
        httpd.server_close()


def _build_layer_data_from_weights(
    model_path: Path,
    max_layers: int = 12,
    *,
    seed: int = 42,
) -> Dict[str, Any]:
    """Build visualization data by reading shapes + stats from real weight files.

    Prefers meta-config.json to recover original architecture parameters.

    Args:
        model_path: Model directory.
        max_layers: Max Transformer blocks to visualize.

    Returns:
        Data dictionary required by the 3D frontend.
    """
    try:
        from vitriol.viz.weight_inspector import generate_viz_data
    except ImportError:
        click.echo(
            "Warning: weight_inspector not available, falling back to config-only mode",
            err=True,
        )
        return _build_layer_data_from_config(model_path, max_layers)

    try:
        viz_data = generate_viz_data(str(model_path), max_layers=max_layers, seed=seed)

        # Convert to the frontend-expected format
        layers_out: List[Dict[str, Any]] = []

        for layer in viz_data.get("layers", []):
            if "sub_layers" in layer:
                # Transformer block: flatten to a frontend-friendly list
                for sl in layer.get("sub_layers", []):
                    entry: Dict[str, Any] = {
                        "name": sl.get("name", ""),
                        "shape": sl.get("shape", []),
                        "type": sl.get("type", "Linear"),
                    }
                    # Inject real stats
                    if sl.get("stats"):
                        entry["stats"] = sl["stats"]
                    layers_out.append(entry)
            else:
                # Standalone layers (embedding, lm_head)
                entry = {
                    "name": layer.get("name", ""),
                    "shape": layer.get("shape", []),
                    "type": layer.get("type", "Linear"),
                }
                if layer.get("stats"):
                    entry["stats"] = layer["stats"]
                layers_out.append(entry)

        result = {
            "model_name": viz_data.get("model_name", model_path.name),
            "hidden_size": viz_data.get("hidden_size", 0),
            "num_layers": viz_data.get("num_layers", 0),
            "vocab_size": viz_data.get("vocab_size", 0),
            "intermediate_size": viz_data.get("intermediate_size", 0),
            "num_heads": viz_data.get("num_attention_heads", 0),
            "config_source": viz_data.get("config_source", "unknown"),
            "weight_stats_available": viz_data.get("weight_stats_available", False),
            "total_params": viz_data.get("total_params", 0),
            "layers": layers_out,
        }

        click.echo(
            f"  Weight data: {len(layers_out)} layers, "
            f"stats={'available' if result['weight_stats_available'] else 'unavailable'}",
            err=True,
        )
        return result

    except Exception as e:
        logger.warning("Failed to read weights, falling back to config-only: %s", e)
        return _build_layer_data_from_config(model_path, max_layers)


def _build_layer_data_from_config(model_path: Path, max_layers: int = 12) -> Dict[str, Any]:
    """Config-derived mode (used when weight files are unavailable)."""
    config_path = model_path / "config.json"
    if not config_path.exists():
        click.echo(f"Error: config.json not found in {model_path}", err=True)
        return {"model_name": model_path.name, "layers": []}

    try:
        config = json.loads(config_path.read_text())
    except Exception as e:
        click.echo(f"Error reading config.json: {e}", err=True)
        return {"model_name": model_path.name, "layers": []}

    # Prefer meta-config.json
    meta = {}
    for meta_name in ("meta-config.json", "config_meta.json"):
        meta_path = model_path / meta_name
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                break
            except Exception:
                logger.debug("Failed to load meta-config for weight visualization")

    effective = meta if meta else config
    text_cfg = effective.get("text_config", effective)

    hidden_size = text_cfg.get("hidden_size") or config.get("hidden_size", 4096)
    num_layers = text_cfg.get("num_hidden_layers") or config.get("num_hidden_layers", 32)
    vocab_size = text_cfg.get("vocab_size") or config.get("vocab_size", 32000)
    intermediate_size = (
        text_cfg.get("intermediate_size")
        or config.get("intermediate_size")
        or hidden_size * 4
    )
    num_heads = text_cfg.get("num_attention_heads") or config.get("num_attention_heads", 32)
    num_kv_heads = text_cfg.get("num_key_value_heads") or config.get("num_key_value_heads", num_heads)
    head_dim = hidden_size // max(num_heads, 1)

    model_name = text_cfg.get("model_type") or effective.get("model_type") or config.get("model_type")
    if not model_name and (effective.get("architectures") or config.get("architectures")):
        model_name = (effective.get("architectures") or config.get("architectures", [None]))[0]
    if not model_name:
        model_name = "Unknown Model"

    data: Dict[str, Any] = {
        "model_name": model_name,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "vocab_size": vocab_size,
        "intermediate_size": intermediate_size,
        "num_heads": num_heads,
        "config_source": "meta-config.json" if meta else "config.json",
        "weight_stats_available": False,
        "layers": [],
    }

    data["layers"].append({
        "name": "embed_tokens",
        "shape": [vocab_size, hidden_size],
        "type": "Embedding",
    })

    for i in range(min(num_layers, max_layers)):
        prefix = f"layers.{i}"
        data["layers"].extend([
            {"name": f"{prefix}.self_attn.q_proj", "shape": [num_heads * head_dim, hidden_size], "type": "Linear"},
            {"name": f"{prefix}.self_attn.k_proj", "shape": [num_kv_heads * head_dim, hidden_size], "type": "Linear"},
            {"name": f"{prefix}.self_attn.v_proj", "shape": [num_kv_heads * head_dim, hidden_size], "type": "Linear"},
            {"name": f"{prefix}.self_attn.o_proj", "shape": [hidden_size, num_heads * head_dim], "type": "Linear"},
            {"name": f"{prefix}.mlp.gate_proj", "shape": [intermediate_size, hidden_size], "type": "Linear"},
            {"name": f"{prefix}.mlp.up_proj", "shape": [intermediate_size, hidden_size], "type": "Linear"},
            {"name": f"{prefix}.mlp.down_proj", "shape": [hidden_size, intermediate_size], "type": "Linear"},
        ])

    data["layers"].append({"name": "lm_head", "shape": [vocab_size, hidden_size], "type": "Linear"})

    return data


@click.command(name="weight-viz")
@click.option('--model-path', '-m', required=True,
              help="Path to the model directory (containing config.json and/or .safetensors/.bin)")
@click.option('--port', default=8781, help="Port for the local 3D weight visualization server")
@click.option('--no-open', is_flag=True, help="Do not open the browser automatically")
@click.option('--config-only', is_flag=True,
              help="Use config-only mode (no weight file reading, faster startup)")
@click.option('--max-layers', default=12, type=int,
              help="Maximum number of Transformer layers to visualize (default: 12)")
@click.option('--seed', default=42, type=int, show_default=True,
              help="Random seed for sampling weight statistics (deterministic)")
def weight_viz(model_path, port, no_open, config_only, max_layers, seed):
    """Visualize model weights in a 3D digital matrix style.

    When real weight files (.safetensors/.bin) are present in the model directory,
    per-layer statistics (mean, std, sparsity, L2 norm, compression ratio) are
    extracted and visualized. Uses meta-config.json to recover original model
    architecture parameters.

    Examples:
        vitriol weight-viz -m ./output/Qwen3.5-397B-Vitriol-ultra-dummy
        vitriol weight-viz -m ./my-model --config-only
        vitriol weight-viz -m ./my-model --max-layers 24
    """
    model_path = Path(model_path)
    if not model_path.exists():
        click.echo(f"Error: Model directory not found: {model_path}", err=True)
        return

    click.echo(f"Extracting model architecture from {model_path}...")

    # Decide whether to read from real weight files
    has_weights = (
        list(model_path.glob("*.safetensors")) or
        list(model_path.glob("*.bin"))
    )

    if config_only or not has_weights:
        if not has_weights and not config_only:
            click.echo("  No weight files found, using config-only mode", err=True)
        data = _build_layer_data_from_config(model_path, max_layers)
    else:
        click.echo("  Found weight files — extracting real statistics...", err=True)
        data = _build_layer_data_from_weights(model_path, max_layers, seed=seed)

    # Write to a temporary directory
    temp_dir = tempfile.mkdtemp()
    json_path = os.path.join(temp_dir, "weight_data.json")
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except Exception as e:
        click.echo(f"Error writing weight_data.json: {e}", err=True)
        return

    # Copy the HTML file
    html_path = Path(__file__).parent.parent.parent / "viz" / "weight_3d_visualizer.html"
    if not html_path.exists():
        click.echo(f"Error: {html_path} not found.", err=True)
        return

    with open(os.path.join(temp_dir, "weight_3d_visualizer.html"), "w", encoding="utf-8") as f:
        f.write(html_path.read_text(encoding="utf-8"))

    click.echo(f"\n  Model: {data.get('model_name', 'Unknown')}")
    click.echo(f"  Config source: {data.get('config_source', 'unknown')}")
    click.echo(f"  Weight stats: {'available' if data.get('weight_stats_available') else 'unavailable'}")
    click.echo(f"  Layers: {len(data.get('layers', []))}")

    serve_3d_weights(temp_dir, port=port, no_open=no_open)
