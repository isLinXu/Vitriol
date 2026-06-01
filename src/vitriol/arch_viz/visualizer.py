
import logging
from pathlib import Path

from .analyzer import ArchitectureAnalyzer
from .parser import ConfigParser
from .renderers.block import BlockRenderer
from .renderers.detail import DetailRenderer
from .renderers.html import HTMLRenderer

logger = logging.getLogger(__name__)

class ArchitectureVisualizer:
    """Main entry point for Architecture Visualization."""

    def __init__(
        self,
        model_id_or_path: str,
        style: str = 'default',
        trust_remote_code: bool = False,
        local_files_only: bool = False,
    ):
        self.model_id = model_id_or_path
        self.style = style
        self.trust_remote_code = bool(trust_remote_code)
        self.local_files_only = bool(local_files_only)
        self.config = None
        self.architecture = None

        # Load and analyze
        self._load()

    def _load(self):
        logger.info(f"Loading config for {self.model_id}...")
        self.config = ConfigParser.load_config(
            self.model_id,
            trust_remote_code=self.trust_remote_code,
            local_files_only=self.local_files_only,
        )
        analyzer = ArchitectureAnalyzer()
        self.architecture = analyzer.analyze(self.config)
        logger.info(f"Analyzed architecture: {self.architecture.model_type} ({self.architecture.total_params/1e9:.2f}B params)")

    def generate_block_diagram(self, output_path: str):
        renderer = BlockRenderer(style=self.style)
        renderer.render(self.architecture, output_path)
        logger.info(f"Generated block diagram: {output_path}")

    def generate_detailed_diagram(self, output_path: str):
        renderer = DetailRenderer()
        renderer.render(self.architecture, output_path)
        logger.info(f"Generated detailed diagram: {output_path}")

    def generate_interactive_html(self, output_path: str):
        renderer = HTMLRenderer()
        renderer.render(self.architecture, output_path)
        logger.info(f"Generated HTML report: {output_path}")

    def generate_all(self, output_dir: str):
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)

        self.architecture.to_json(path / "architecture.json")
        self.generate_block_diagram(str(path / "block_diagram.png"))
        self.generate_detailed_diagram(str(path / "detailed_view.png"))
        self.generate_interactive_html(str(path / "interactive.html"))
