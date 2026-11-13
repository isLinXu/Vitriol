
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

class IncrementalGenerator:
    """Support incremental generation and checkpointing"""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.checkpoint_file = Path(output_dir) / '.vitriol_checkpoint.json'

    def save_checkpoint(self, state: Dict[str, Any], *, strict: bool = False) -> None:
        """Save progress atomically; strict mode propagates commit failures."""
        temporary_path = None
        try:
            self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", dir=self.checkpoint_file.parent,
                                             prefix=".vitriol-checkpoint-", delete=False) as f:
                temporary_path = f.name
                json.dump(state, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary_path, self.checkpoint_file)
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")
            if strict:
                raise
        finally:
            if temporary_path is not None and os.path.exists(temporary_path):
                os.unlink(temporary_path)

    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load previous progress"""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file) as f:
                    state = json.load(f)
                if not isinstance(state, dict):
                    raise ValueError("checkpoint must be a JSON object")
                return state
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
