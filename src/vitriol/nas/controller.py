import json
import logging
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional

from .evaluator import HybridEvaluator
from .rl_agent import RLSearcher
from .search_space import ArchitectureGene, LLMSearchSpace
from .searcher import EvolutionarySearcher, RandomSearcher

logger = logging.getLogger(__name__)

class NASController:
    """Orchestrates the Neural Architecture Search process."""

    def __init__(self, output_dir: str = "output/nas_results", device: str = "cpu"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.search_space = LLMSearchSpace()
        self.evaluator = HybridEvaluator(str(self.output_dir / "eval"), device=device)
        self.checkpoint_path = self.output_dir / "checkpoint.json"

    def run(self, algorithm: str = "random", n_iterations: int = 10, population_size: int = 20, resume: bool = False, dataset_config: Optional[Dict] = None) -> Dict[str, Any]:
        """Run the search."""
        logger.info("Starting NAS run with algorithm=%s, iterations=%d", algorithm, n_iterations)

        searcher = None

        # Define checkpoint callback
        def checkpoint_cb() -> None:
            if searcher:
                self._save_checkpoint(searcher)

        if algorithm == "random":
            searcher = RandomSearcher(self.search_space, self.evaluator, save_checkpoint_callback=checkpoint_cb)
        elif algorithm == "evolutionary":
            searcher = EvolutionarySearcher(
                self.search_space,
                self.evaluator,
                population_size=population_size,
                save_checkpoint_callback=checkpoint_cb
            )
        elif algorithm == "rl":
            searcher = RLSearcher(
                self.search_space,
                self.evaluator,
                device=str(self.evaluator.device),
                save_checkpoint_callback=checkpoint_cb,
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        # Inject dataset config into evaluator
        original_evaluate = self.evaluator.evaluate
        if dataset_config:
            logger.info("Using dataset config: %s", dataset_config)
            self.evaluator.evaluate = partial(original_evaluate, dataset_config=dataset_config)

        # Resume logic
        if resume and self.checkpoint_path.exists():
            logger.info("Resuming from checkpoint: %s", self.checkpoint_path)
            self._load_checkpoint(searcher)

        try:
            best_gene = searcher.search(n_iterations)

        except KeyboardInterrupt:
            logger.info("Search interrupted! Saving checkpoint...")
            self._save_checkpoint(searcher)
            raise
        finally:
            self.evaluator.evaluate = original_evaluate

        # Save final results
        results = {
            "best_gene": best_gene.to_config() if best_gene else None,
            "history": [
                {
                    "gene": r["gene"].to_config(),
                    "score": r["score"],
                    "metrics": r["result"]
                }
                for r in searcher.history
            ]
        }

        output_file = self.output_dir / "results.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        # Clean checkpoint on success
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()

        logger.info("NAS run completed. Results saved to %s", output_file)
        return results

    def _save_checkpoint(self, searcher):
        """Save search state to file."""
        # Convert history to serializable format
        history_serializable = [
            {
                "gene": r["gene"].to_dict(),
                "score": r["score"],
                "metrics": r["result"]
            }
            for r in searcher.history
        ]

        data = {
            "history": history_serializable,
        }

        # If Evolutionary, save population
        if hasattr(searcher, "population"):
            data["population"] = [g.to_dict() for g in searcher.population]

        with open(self.checkpoint_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Checkpoint saved.")

    def _load_checkpoint(self, searcher):
        """Load search state from file."""
        if not self.checkpoint_path.exists():
            return

        try:
            with open(self.checkpoint_path) as f:
                data = json.load(f)

            # Restore history
            if "history" in data:
                searcher.history = []
                for r in data["history"]:
                    gene = ArchitectureGene.from_dict(r["gene"])
                    searcher.history.append({
                        "gene": gene,
                        "score": r["score"],
                        "result": r["metrics"]
                    })
                logger.info("Restored %d history records.", len(searcher.history))

            # Restore population
            if "population" in data and hasattr(searcher, "population"):
                searcher.population = [ArchitectureGene.from_dict(g) for g in data["population"]]
                logger.info("Restored population of size %d.", len(searcher.population))

        except Exception as e:
            logger.error("Failed to load checkpoint: %s", e)
            # Don't crash, just start fresh if checkpoint is corrupted
