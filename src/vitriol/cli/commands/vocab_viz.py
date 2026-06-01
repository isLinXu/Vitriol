import json
import logging
import os
import tempfile
import threading
import time
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import click

logger = logging.getLogger(__name__)


def _missing_viz_dependency(exc: Exception) -> click.ClickException:
    return click.ClickException(
        f"Missing optional visualization dependency ({exc}). Install it with: pip install -e '.[viz]' (package extra: vitriol[viz])"
    )

def _load_vocab_from_local_tokenizer_files(model_dir: str) -> tuple[dict[str, int], set[str]]:
    model_path = Path(model_dir)
    tok_json = model_path / "tokenizer.json"
    if not tok_json.exists():
        raise FileNotFoundError(f"tokenizer.json not found under {model_dir}")

    with tok_json.open("r", encoding="utf-8") as f:
        tok = json.load(f)

    model = tok.get("model", {})
    vocab = model.get("vocab", None)
    if not isinstance(vocab, dict):
        raise ValueError("Unsupported tokenizer.json: missing model.vocab dict")

    special_tokens: set[str] = set()
    tok_cfg_path = model_path / "tokenizer_config.json"
    if tok_cfg_path.exists():
        with tok_cfg_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)

        for k in (
            "bos_token",
            "eos_token",
            "pad_token",
            "unk_token",
            "sep_token",
            "cls_token",
            "mask_token",
            "audio_bos_token",
            "audio_eos_token",
            "audio_token",
            "image_token",
            "video_token",
            "vision_bos_token",
            "vision_eos_token",
        ):
            v = cfg.get(k)
            if isinstance(v, str) and v:
                special_tokens.add(v)

        msst = cfg.get("model_specific_special_tokens", {})
        if isinstance(msst, dict):
            for v in msst.values():
                if isinstance(v, str) and v:
                    special_tokens.add(v)

        ast = cfg.get("additional_special_tokens")
        if isinstance(ast, list):
            for v in ast:
                if isinstance(v, str) and v:
                    special_tokens.add(v)

    return vocab, special_tokens

class QuietHTTPHandler(SimpleHTTPRequestHandler):
    """HTTP handler that suppresses logging"""
    def log_message(self, format, *args):
        pass  # Suppress request logging

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

def serve_3d_vocab(temp_dir: str):
    port = 8780
    max_retries = 50
    for _ in range(max_retries):
        try:
            httpd = HTTPServer(('127.0.0.1', port), lambda *args, **kwargs: QuietHTTPHandler(*args, directory=temp_dir, **kwargs))
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
            httpd = HTTPServer(('127.0.0.1', 0), lambda *args, **kwargs: QuietHTTPHandler(*args, directory=temp_dir, **kwargs))
            port = httpd.server_port
            click.echo(f"OS assigned port {port}.")
        except OSError as e:
            click.echo(f"Error: Could not start server. {e}", err=True)
            return

    url = f"http://127.0.0.1:{port}/vocab_3d_visualizer.html"
    click.echo(f"\nStarting 3D Vocab Visualizer at: {url}")
    click.echo("Press Ctrl+C to stop the server.")

    threading.Thread(target=lambda: (time.sleep(1), webbrowser.open(url)), daemon=True).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        click.echo("\nShutting down server...")
    finally:
        httpd.server_close()

@click.command()
@click.option('--output', '-o', default="output/vocab_viz.html", help="Output path for the HTML visualization")
@click.option('--type', '-t', 'viz_type', type=click.Choice(["treemap", "bar", "single"]), default="treemap", help="Type of visualization")
@click.option('--plot-type', '-p', type=click.Choice(["treemap", "length-hist", "first-char", "compression-radar", "unicode-sunburst", "vocab-map", "digit-coverage", "subword-fertility", "special-tokens"]), default="treemap", help="Plot type for single mode")
@click.option('--model-id', '-m', help="Add a specific HF model/tokenizer to the visualization (Required for --type single or --3d)")
@click.option('--3d', 'is_3d', is_flag=True, help="Launch the interactive 3D vocabulary visualizer in browser")
def vocab_viz(output, viz_type, model_id, plot_type, is_3d):
    """Visualize tokenizer vocabulary sizes."""
    if is_3d:
        from ...utils.hf_loading import load_tokenizer as hf_load_tokenizer

        if not model_id:
            logger.error("--model-id is required for 3D visualization.")
            return

        click.echo(f"Loading tokenizer for {model_id}...")
        try:
            ctx = click.get_current_context(silent=True)
            trust_remote_code = bool((ctx.obj or {}).get("trust_remote_code", False)) if ctx else False
            allow_network = bool((ctx.obj or {}).get("allow_network", True)) if ctx else True
            local_files_only = bool((ctx.obj or {}).get("local_files_only", False)) if ctx else False
            tokenizer = hf_load_tokenizer(
                model_id,
                security={
                    "trust_remote_code": trust_remote_code,
                    "allow_network": allow_network,
                    "local_files_only": local_files_only,
                },
            )
            vocab = tokenizer.get_vocab()
            special_ids = set(tokenizer.all_special_ids)
        except Exception as e:
            try:
                vocab, special_tokens = _load_vocab_from_local_tokenizer_files(model_id)
                special_ids = {int(vocab[t]) for t in special_tokens if t in vocab}
                click.echo("Loaded tokenizer vocabulary from local tokenizer.json.")
            except Exception:
                click.echo(f"Error loading tokenizer: {e}", err=True)
                return

        click.echo("Processing vocabulary and assigning categories...")
        categories = ["Special", "English/Latin", "Chinese", "Digits", "Cyrillic", "Other/Symbol"]
        cat_map = {c: i for i, c in enumerate(categories)}

        sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
        tokens_data = []

        for token, idx in sorted_vocab:
            # pad missing IDs
            while len(tokens_data) < idx:
                tokens_data.append(["<unk>", cat_map["Special"]])

            clean_token = token.replace('Ġ', '').replace(' ', '').replace('##', '').replace(' ', '')
            cat = "Other/Symbol"

            if idx in special_ids:
                cat = "Special"
            else:
                if not clean_token:
                    cat = "Special"
                elif all(c.isascii() and c.isalpha() for c in clean_token):
                    cat = "English/Latin"
                elif all(c.isdigit() for c in clean_token):
                    cat = "Digits"
                elif any('\u4e00' <= c <= '\u9fff' for c in clean_token):
                    cat = "Chinese"
                elif any('\u0400' <= c <= '\u04FF' for c in clean_token):
                    cat = "Cyrillic"
                else:
                    cat = "Other/Symbol"

            tokens_data.append([token, cat_map[cat]])

        data = {
            "model_id": model_id,
            "vocab_size": len(tokens_data),
            "categories": categories,
            "tokens": tokens_data
        }

        # Write to temp directory and serve
        temp_dir = tempfile.mkdtemp()
        json_path = os.path.join(temp_dir, "vocab_data.json")
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, separators=(',', ':'))
            click.echo(f"Successfully wrote {json_path} ({os.path.getsize(json_path)} bytes)")
        except Exception as e:
            click.echo(f"Error writing vocab_data.json: {e}", err=True)
            return

        # Copy HTML
        html_path = Path(__file__).parent.parent.parent / "viz" / "vocab_3d_visualizer.html"
        with open(os.path.join(temp_dir, "vocab_3d_visualizer.html"), "w", encoding="utf-8") as f:
            f.write(html_path.read_text(encoding="utf-8"))

        serve_3d_vocab(temp_dir)
        return

    logger.info(f"Generating Vocab Visualization ({viz_type})...")

    try:
        from vitriol.vocab_viz.core import VocabVisualizer
    except (ImportError, ModuleNotFoundError) as exc:
        raise _missing_viz_dependency(exc) from exc

    viz = VocabVisualizer() # Uses default dataset

    if viz_type == "single":
        if not model_id:
            logger.error("--model-id is required for single mode.")
            return
        path = viz.generate_single_distribution(model_id, output, plot_type=plot_type)
        if path:
             logger.info(f"Visualization saved to {path}")
        return

    if model_id:
        viz.add_model_from_id(model_id, family="Custom")

    if viz_type == "treemap":
        path = viz.generate_treemap(output)
    else:
        path = viz.generate_bar_chart(output)

    logger.info(f"Visualization saved to {path}")
