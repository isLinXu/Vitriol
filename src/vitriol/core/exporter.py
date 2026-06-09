
import json
import logging
from pathlib import Path

from ..utils.hf_loading import load_config_or_raw

logger = logging.getLogger(__name__)

class ModelExporter:
    """Export model information and formats"""

    def __init__(self, input_dir: str, trust_remote_code: bool = False):
        self.input_dir = Path(input_dir)
        self.trust_remote_code = trust_remote_code

    def _load_best_config(self):
        """Load the best available config: meta-config.json > config_meta.json > config.json.

        meta-config.json contains the original unmodified HuggingFace config,
        preserving the real model architecture even when config.json is shrunk.
        """
        import tempfile

        for meta_name in ('meta-config.json', 'config_meta.json'):
            meta_path = self.input_dir / meta_name
            if meta_path.exists():
                try:
                    meta_dict = json.loads(meta_path.read_text())
                    with tempfile.TemporaryDirectory() as tmp:
                        (Path(tmp) / "config.json").write_text(json.dumps(meta_dict, indent=2))
                        config = load_config_or_raw(
                            tmp,
                            security={
                                "trust_remote_code": self.trust_remote_code,
                                "allow_network": False,
                                "local_files_only": True,
                            },
                        )
                        logger.info(f"Loaded original config from {meta_name}")
                        return config
                except Exception as e:
                    logger.warning(f"Failed to load {meta_name}: {e}")

        return load_config_or_raw(
            str(self.input_dir),
            security={
                "trust_remote_code": self.trust_remote_code,
                "allow_network": False,
                "local_files_only": True,
            },
        )

    @staticmethod
    def _resolve_gguf_output_path(output_target: str) -> Path:
        """Resolve GGUF output target.

        The CLI exposes ``--output`` as a file path, but older internal callers may
        still pass a directory. We support both:
        - explicit file path (has a suffix like ``.gguf``) -> use as-is
        - directory path (existing dir or suffix-less path) -> write ``model.gguf`` inside
        """
        target = Path(output_target)
        if target.exists() and target.is_dir():
            gguf_path = target / "model.gguf"
        elif target.suffix:
            gguf_path = target
        else:
            gguf_path = target / "model.gguf"
        gguf_path.parent.mkdir(parents=True, exist_ok=True)
        return gguf_path

    def export_structure(self, output_file: str) -> None:
        """Export architecture details to JSON"""
        try:
            config = self._load_best_config()

            structure = {
                'model_type': getattr(config, 'model_type', 'unknown'),
                'architectures': getattr(config, 'architectures', []),
                'hidden_size': getattr(config, 'hidden_size', None),
                'num_layers': getattr(config, 'num_hidden_layers', None),
                'num_heads': getattr(config, 'num_attention_heads', None),
                'vocab_size': getattr(config, 'vocab_size', None),
                'config': config.to_dict()
            }

            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(structure, indent=2), encoding="utf-8")

            logger.info(f"Exported structure to {output_file}")

        except Exception as e:
            logger.error(f"Failed to export structure: {e}")
            raise e

    def export_gguf_prep(self, output_target: str) -> None:
        """Run GGUF conversion using llama.cpp if available.

        ``output_target`` may be either a final ``.gguf`` file path or a directory.
        """
        import subprocess
        import sys

        gguf_path = self._resolve_gguf_output_path(output_target)

        logger.info(f"Attempting GGUF export to {gguf_path}...")

        # Try to use installed llama-cpp-python or external script
        # Assuming llama-cpp-python is installed which provides a conversion script
        # Or check for 'convert-hf-to-gguf.py' in PATH

        cmd = [sys.executable, "-m", "llama_cpp.convert_hf_to_gguf", str(self.input_dir), "--outfile", str(gguf_path)]

        try:
            logger.info(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            logger.info("GGUF export successful!")
        except Exception as e:
            logger.warning(f"Automated GGUF export failed: {e}")
            logger.warning("Please install llama-cpp-python or use the official conversion script:")
            logger.warning(f"python convert_hf_to_gguf.py {self.input_dir} --outfile {gguf_path}")
