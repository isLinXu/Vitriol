import logging
import random
from typing import List, Dict, Any
from tqdm import tqdm

from .search_space import SearchSpace, ArchitectureGene, LLMSearchSpace
from .evaluator import HybridEvaluator

logger = logging.getLogger(__name__)


class Searcher:
    """Base class for search algorithms."""

    def __init__(self, search_space: SearchSpace, evaluator: HybridEvaluator, save_checkpoint_callback=None):
        self.search_space = search_space
        self.evaluator = evaluator
        self.history: List[Dict[str, Any]] = []
        self.save_checkpoint_callback = save_checkpoint_callback

    def search(self, n_iterations: int) -> ArchitectureGene:
        raise NotImplementedError

    def _evaluate(self, gene: ArchitectureGene) -> float:
        result = self.evaluator.evaluate(gene)
        score = result["score"]

        record = {
            "gene": gene,
            "result": result,
            "score": score,
        }
        self.history.append(record)

        if self.save_checkpoint_callback:
            try:
                self.save_checkpoint_callback()
            except Exception as e:
                logger.warning(f"Failed to save checkpoint: {e}")

        return score


class RandomSearcher(Searcher):
    """Random Search Algorithm."""

    def search(self, n_iterations: int) -> ArchitectureGene:
        best_gene = None
        best_score = -float("inf")

        logger.info(f"Starting Random Search for {n_iterations} iterations...")

        for i in tqdm(range(n_iterations), desc="Random Search"):
            gene = self.search_space.sample()
            score = self._evaluate(gene)

            if score > best_score:
                best_score = score
                best_gene = gene
                logger.info(f"New best score: {best_score:.4f} (Iter {i})")

        return best_gene


class EvolutionarySearcher(Searcher):
    """Evolutionary Algorithm (Genetic Algorithm)."""

    def __init__(self, search_space: LLMSearchSpace, evaluator: HybridEvaluator,
                 population_size: int = 20, mutation_rate: float = 0.1, save_checkpoint_callback=None):
        super().__init__(search_space, evaluator, save_checkpoint_callback)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population: List[ArchitectureGene] = []

    def search(self, n_generations: int) -> ArchitectureGene:
        if not self.population:
            self.population = [self.search_space.sample() for _ in range(self.population_size)]

        best_gene = None
        best_score = -float("inf")

        if self.history:
            best_record = max(self.history, key=lambda item: item["score"])
            best_gene = best_record["gene"]
            best_score = best_record["score"]

        logger.info(f"Starting Evolutionary Search: {n_generations} generations, pop_size={self.population_size}")

        for gen in range(n_generations):
            logger.info(f"Generation {gen + 1}/{n_generations}")

            scores = []
            for gene in tqdm(self.population, desc=f"Evaluating Gen {gen + 1}"):
                score = self._evaluate(gene)
                scores.append(score)

                if score > best_score:
                    best_score = score
                    best_gene = gene
                    logger.info(f"New best score: {best_score:.4f}")

            parents = self._selection(self.population, scores, k=self.population_size // 2)

            offspring = []
            while len(offspring) < self.population_size - len(parents):
                p1, p2 = random.sample(parents, 2)
                child = self._crossover(p1, p2)
                child = self.search_space.mutate(child, self.mutation_rate)
                offspring.append(child)

            self.population = parents + offspring
            if self.save_checkpoint_callback:
                try:
                    self.save_checkpoint_callback()
                except Exception as e:
                    logger.warning(f"Failed to save checkpoint after generation {gen + 1}: {e}")

        return best_gene

    def _selection(self, population: List[ArchitectureGene], scores: List[float], k: int) -> List[ArchitectureGene]:
        """Select top-k individuals based on fitness."""
        combined = list(zip(population, scores))
        combined.sort(key=lambda x: x[1], reverse=True)
        return [g for g, _ in combined[:k]]

    def _crossover(self, p1: ArchitectureGene, p2: ArchitectureGene) -> ArchitectureGene:
        """Uniform crossover."""
        import dataclasses
        fields = dataclasses.fields(p1)

        new_kwargs = {}
        for f in fields:
            if not f.init:
                continue
            val = getattr(p1, f.name) if random.random() < 0.5 else getattr(p2, f.name)
            new_kwargs[f.name] = val

        return ArchitectureGene(**new_kwargs)
