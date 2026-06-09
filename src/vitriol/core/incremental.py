
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

class IncrementalGenerator:
    """Support incremental generation and checkpointing"""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.checkpoint_file = Path(output_dir) / '.vitriol_checkpoint.json'

    def save_checkpoint(self, state: Dict[str, Any]) -> None:
        """Save generation progress"""
        try:
            self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.checkpoint_file, 'w') as f:
                json.dump(state, f)
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")

    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load previous progress"""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
                return None
        return None

    def clear_checkpoint(self) -> None:
        """Remove checkpoint file after completion"""
        if self.checkpoint_file.exists():
            try:
                self.checkpoint_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to remove checkpoint: {e}")
