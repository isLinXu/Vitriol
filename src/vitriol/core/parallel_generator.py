"""
Parallel weight generation for improved performance.

This module provides utilities for generating model weights in parallel,
significantly speeding up the generation process for large models.
"""

import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

from ..strategies.base import WeightGenerationStrategy

logger = logging.getLogger(__name__)


class ParallelWeightGenerator:
    """
    Generate weights in parallel for speed improvement.

    This class uses multi-threading or multi-processing to generate
    multiple weight tensors simultaneously, which can significantly
    speed up generation for models with many parameters.

    Example:
        >>> generator = ParallelWeightGenerator(strategy, n_workers=4)
        >>> weights = generator.generate_shard(param_names, param_shapes, param_dtypes)
    """

    def __init__(
        self,
        strategy: WeightGenerationStrategy,
        n_workers: Optional[int] = None,
        use_processes: bool = False
    ):
        """
        Initialize parallel generator.

        Args:
            strategy: Weight generation strategy to use
            n_workers: Number of parallel workers (default: CPU count)
            use_processes: Use processes instead of threads (for CPU-bound tasks)
        """
        self.strategy = strategy
        self.n_workers = n_workers or mp.cpu_count()
        self.use_processes = use_processes

        logger.info(
            f"ParallelWeightGenerator initialized with {self.n_workers} workers "
            f"(mode: {'processes' if use_processes else 'threads'})"
        )

    def generate_shard_parallel(
        self,
        param_names: List[str],
        param_shapes: Dict[str, tuple],
        param_dtypes: Dict[str, torch.dtype],
        show_progress: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Generate tensors for a shard in parallel.

        Args:
            param_names: List of parameter names to generate
            param_shapes: Dict mapping parameter names to shapes
            param_dtypes: Dict mapping parameter names to dtypes
            show_progress: Whether to show progress bar

        Returns:
            Dict mapping parameter names to generated tensors
        """
        # Choose executor based on task type
        ExecutorClass = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor

        results = {}

        with ExecutorClass(max_workers=self.n_workers) as executor:
            # Submit all tasks
            future_to_name = {
                executor.submit(
                    self._generate_single_tensor,
                    name,
                    param_shapes[name],
                    param_dtypes[name]
                ): name
                for name in param_names
            }

            # Collect results with progress bar
            if show_progress:
                pbar = tqdm(
                    total=len(future_to_name),
                    desc="Generating weights",
                    unit="tensor"
                )
            else:
                pbar = None

            for future in as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    tensor = future.result()
                    results[name] = tensor
                except Exception as e:
                    logger.error(f"Failed to generate tensor '{name}': {e}")
                    raise

                if pbar:
                    pbar.update(1)

            if pbar:
                pbar.close()

        return results

    def _generate_single_tensor(
        self,
        name: str,
        shape: tuple,
        dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Generate a single tensor (wrapper for strategy).

        Args:
            name: Parameter name
            shape: Tensor shape
            dtype: Data type

        Returns:
            Generated tensor
        """
        return self.strategy.generate_tensor(shape, dtype, name)

    def generate_batch_parallel(
        self,
        batches: List[Dict[str, Dict]],
        show_progress: bool = True
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Generate multiple batches of tensors in parallel.

        Args:
            batches: List of batch dicts, each containing:
                {
                    "param_name": {"shape": ..., "dtype": ...},
                    ...
                }
            show_progress: Whether to show progress bar

        Returns:
            List of generated tensor dicts
        """
        all_results = []

        for batch_idx, batch in enumerate(batches):
            logger.info(f"Generating batch {batch_idx + 1}/{len(batches)}")

            param_names = list(batch.keys())
            param_shapes = {name: info["shape"] for name, info in batch.items()}
            param_dtypes = {name: info["dtype"] for name, info in batch.items()}

            results = self.generate_shard_parallel(
                param_names,
                param_shapes,
                param_dtypes,
                show_progress=show_progress
            )

            all_results.append(results)

        return all_results


class StreamingWeightGenerator:
    """
    Generate weights in streaming fashion to minimize memory usage.

    This class generates and saves weights one shard at a time,
    avoiding loading all weights into memory simultaneously.

    Example:
        >>> generator = StreamingWeightGenerator(strategy, shard_manager)
        >>> generator.generate_streaming(model, output_dir)
    """

    def __init__(
        self,
        strategy: WeightGenerationStrategy,
        shard_manager,
        n_workers: Optional[int] = None
    ):
        """
        Initialize streaming generator.

        Args:
            strategy: Weight generation strategy
            shard_manager: Shard manager for planning splits
            n_workers: Number of parallel workers (optional)
        """
        self.strategy = strategy
        self.shard_manager = shard_manager
        self.parallel_gen = ParallelWeightGenerator(strategy, n_workers) if n_workers else None

    def generate_streaming(
        self,
        model: torch.nn.Module,
        output_dir: str,
        show_progress: bool = True
    ) -> List[str]:
        """
        Generate and save weights in streaming fashion.

        Args:
            model: PyTorch model
            output_dir: Output directory for weight files
            show_progress: Whether to show progress

        Returns:
            List of saved shard filenames
        """
        from pathlib import Path

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_shards = []

        # Get parameter info
        param_info = self._get_param_info(model)

        # Generate and save shards one by one
        if show_progress:
            pbar = tqdm(
                self.shard_manager.plan_shards(
                    param_info["names"],
                    param_info["sizes"]
                ),
                desc="Generating shards",
                unit="shard"
            )
        else:
            pbar = None

        iterator = pbar if pbar else self.shard_manager.plan_shards(
            param_info["names"],
            param_info["sizes"]
        )

        for shard_name, param_batch in iterator:
            # Generate tensors for this shard
            shard_data = self._generate_shard_data(
                param_batch,
                param_info["shapes"],
                param_info["dtypes"]
            )

            # Save shard
            shard_path = output_path / shard_name
            self.strategy.save_shard(shard_data, str(shard_path))
            saved_shards.append(shard_name)

            # Clear memory
            del shard_data

        if pbar:
            pbar.close()

        return saved_shards

    def _get_param_info(self, model: torch.nn.Module) -> Dict:
        """
        Extract parameter information from model.

        Args:
            model: PyTorch model

        Returns:
            Dict with names, shapes, dtypes, sizes
        """
        names = []
        shapes = {}
        dtypes = {}
        sizes = {}

        # Determine dtype size
        dtype_size = 2  # bfloat16 default

        for name, param in model.named_parameters():
            names.append(name)
            shapes[name] = param.shape
            dtypes[name] = param.dtype
            sizes[name] = param.numel() * dtype_size

        return {
            "names": names,
            "shapes": shapes,
            "dtypes": dtypes,
            "sizes": sizes
        }

    def _generate_shard_data(
        self,
        param_batch: Dict[str, int],
        shapes: Dict[str, tuple],
        dtypes: Dict[str, torch.dtype]
    ) -> Dict[str, torch.Tensor]:
        """
        Generate tensor data for a shard.

        Args:
            param_batch: Dict of param_name -> size
            shapes: Dict of param_name -> shape
            dtypes: Dict of param_name -> dtype

        Returns:
            Dict of param_name -> tensor
        """
        if self.parallel_gen:
            return self.parallel_gen.generate_shard_parallel(
                list(param_batch.keys()),
                shapes,
                dtypes,
                show_progress=False
            )
        else:
            # Sequential generation
            return {
                name: self.strategy.generate_tensor(
                    shapes[name],
                    dtypes[name],
                    name
                )
                for name in param_batch.keys()
            }
