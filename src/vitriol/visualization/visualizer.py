
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple
import plotly.express as px
from scipy.stats import entropy
import logging
import math

logger = logging.getLogger(__name__)

class WeightVisualizer:
    def __init__(
        self,
        figsize: Tuple[int, int] = (12, 8),
        style: str = 'seaborn-v0_8',
        *,
        seed: int = 42,
        sample_size: int = 1_000_000,
    ):
        self.figsize = figsize
        self.seed = int(seed)
        self.sample_size = int(sample_size)
        try:
            plt.style.use(style)
        except OSError:
            # Fallback if style not found
            plt.style.use('default')
            
        self.colors = sns.color_palette("husl", 8)

    @staticmethod
    def _tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
        """Safely materialize a tensor for visualization.

        Visualization utilities should work for tensors that live on accelerators
        or still require gradients from upstream experiments.
        """
        return tensor.detach().float().cpu().numpy()

    def _flatten_weights(self, weights: Dict[str, torch.Tensor]) -> np.ndarray:
        """Flatten all weights into a single numpy array"""
        ordered_items = [(name, weights[name]) for name in sorted(weights.keys())]
        if not ordered_items:
            return np.array([], dtype=np.float32)

        # P0: reproducibility: stable iteration order + deterministic sampling with a fixed seed
        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.seed)

        sample_counts = []
        total_values = 0
        for _, param in ordered_items:
            count = min(int(param.numel()), self.sample_size) if param.numel() > 0 else 0
            sample_counts.append(count)
            total_values += count

        if total_values == 0:
            return np.array([], dtype=np.float32)

        flattened = np.empty(total_values, dtype=np.float32)
        offset = 0

        for (_, param), count in zip(ordered_items, sample_counts):
            if count <= 0:
                continue

            flat = param.detach().reshape(-1)
            # Sample large tensors to avoid OOM
            if flat.numel() > self.sample_size:
                # Deterministic uniform sampling (with replacement) to avoid randperm cost
                indices = torch.randint(0, flat.numel(), (self.sample_size,), generator=gen)
                values = self._tensor_to_numpy(flat.index_select(0, indices.to(device=flat.device)))
            else:
                values = self._tensor_to_numpy(flat)

            end = offset + values.shape[0]
            flattened[offset:end] = values
            offset = end

        return flattened

    def visualize_weight_distribution(self, weights: Dict[str, torch.Tensor], title: str = "Weight Distribution"):
        """1. Weight Distribution Histogram"""
        flat_weights = self._flatten_weights(weights)
        if flat_weights.size == 0:
            logger.warning("No weights to visualize")
            return None

        fig, ax = plt.subplots(figsize=self.figsize)
        sns.histplot(flat_weights, kde=True, ax=ax, bins=50)
        
        mean = np.mean(flat_weights)
        std = np.std(flat_weights)
        unique_count = len(np.unique(flat_weights))
        
        ax.set_title(f"{title}\nμ={mean:.4f}, σ={std:.4f}, Unique={unique_count}")
        ax.set_xlabel("Weight Value")
        ax.set_ylabel("Frequency")
        
        return fig

    def visualize_weight_heatmap(self, weights: Dict[str, torch.Tensor], layer_name: Optional[str] = None):
        """2. Weight Matrix Heatmap"""
        # Select a representative layer (2D)
        target_w = None
        target_name = ""
        
        if layer_name and layer_name in weights:
            target_w = weights[layer_name]
            target_name = layer_name
        else:
            # Find first 2D weight
            for name, param in weights.items():
                if param.dim() == 2:
                    target_w = param
                    target_name = name
                    break
        
        if target_w is None:
            logger.warning("No 2D weight matrix found for heatmap")
            return None

        # Downsample if too large for heatmap
        w_np = self._tensor_to_numpy(target_w)
        rows, cols = w_np.shape
        if rows > 1000 or cols > 1000:
            # Simple slicing for visualization
            w_np = w_np[:min(rows, 500), :min(cols, 500)]
            target_name += " (Top-Left 500x500)"

        fig, ax = plt.subplots(figsize=self.figsize)
        sns.heatmap(w_np, cmap="RdBu_r", center=0, ax=ax)
        ax.set_title(f"Weight Heatmap: {target_name}")
        
        return fig

    def visualize_sparsity_pattern(self, weights: Dict[str, torch.Tensor], layer_name: Optional[str] = None):
        """3. Sparsity Pattern"""
        # Similar selection logic as heatmap
        target_w = None
        target_name = ""
        
        if layer_name and layer_name in weights:
            target_w = weights[layer_name]
            target_name = layer_name
        else:
            for name, param in weights.items():
                if param.dim() == 2:
                    target_w = param
                    target_name = name
                    break
        
        if target_w is None:
            return None

        w_np = self._tensor_to_numpy(target_w)
        rows, cols = w_np.shape
        if rows > 1000 or cols > 1000:
            w_np = w_np[:min(rows, 500), :min(cols, 500)]
            target_name += " (Top-Left 500x500)"

        # Binary mask: 0 is white, non-zero is black
        mask = (w_np != 0).astype(int)
        sparsity = 1.0 - (np.count_nonzero(w_np) / w_np.size)

        fig, ax = plt.subplots(figsize=self.figsize)
        ax.imshow(mask, cmap="binary", interpolation="nearest")
        ax.set_title(f"Sparsity Pattern: {target_name}\nSparsity: {sparsity:.2%}")
        ax.set_xlabel("Output Dim")
        ax.set_ylabel("Input Dim")
        
        return fig

    def visualize_value_frequency(self, weights: Dict[str, torch.Tensor], top_k: int = 20):
        """4. Value Frequency Analysis"""
        flat_weights = self._flatten_weights(weights)
        if flat_weights.size == 0:
            return None

        # Round to avoid float precision issues for "unique" check
        rounded = np.round(flat_weights, decimals=6)
        unique, counts = np.unique(rounded, return_counts=True)
        
        # Sort by frequency
        sorted_indices = np.argsort(-counts)
        top_indices = sorted_indices[:top_k]
        
        top_values = unique[top_indices]
        top_counts = counts[top_indices]
        
        fig, ax = plt.subplots(figsize=self.figsize)
        bars = ax.bar(range(len(top_values)), top_counts)
        ax.set_xticks(range(len(top_values)))
        ax.set_xticklabels([f"{v:.4g}" for v in top_values], rotation=45)
        
        ax.set_title(f"Top {top_k} Frequent Values")
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        
        # Add labels
        for rect in bars:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
                    
        return fig

    def visualize_statistical_comparison(self, weights_dict: Dict[str, Dict[str, torch.Tensor]]):
        """5. Statistical Comparison (Boxplot etc.) for multiple strategies"""
        stats_data = []
        
        for strategy_name, w_dict in weights_dict.items():
            flat = self._flatten_weights(w_dict)
            if flat.size == 0:
                continue
                
            stats_data.append({
                "Strategy": strategy_name,
                "Mean": np.mean(flat),
                "Std": np.std(flat),
                "Min": np.min(flat),
                "Max": np.max(flat),
                "Sparsity": 1.0 - (np.count_nonzero(flat) / flat.size),
                "UniqueRatio": len(np.unique(flat)) / flat.size
            })
            
        df = pd.DataFrame(stats_data)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        sns.barplot(data=df, x="Strategy", y="Mean", ax=axes[0,0])
        axes[0,0].set_title("Mean Value")
        
        sns.barplot(data=df, x="Strategy", y="Std", ax=axes[0,1])
        axes[0,1].set_title("Standard Deviation")
        
        sns.barplot(data=df, x="Strategy", y="Sparsity", ax=axes[1,0])
        axes[1,0].set_title("Sparsity Ratio")
        
        sns.barplot(data=df, x="Strategy", y="UniqueRatio", ax=axes[1,1])
        axes[1,1].set_title("Unique Value Ratio")
        
        plt.tight_layout()
        return fig

    def visualize_compression_potential(self, weights: Dict[str, torch.Tensor]):
        """6. Compression Potential Analysis"""
        flat = self._flatten_weights(weights)
        if flat.size == 0:
            return None
            
        # 1. Entropy
        # Discretize for entropy calculation
        hist, _ = np.histogram(flat, bins=256, density=True)
        # Remove zeros for log
        hist = hist[hist > 0]
        ent = entropy(hist, base=2)
        
        # 2. Unique Ratio
        n_unique = len(np.unique(flat))
        unique_ratio = n_unique / flat.size
        
        # 3. Effective Bits
        effective_bits = math.log2(n_unique) if n_unique > 0 else 0
        
        # 4. Estimated Compression Ratio (Naive)
        # Assuming float32 (32 bits) -> effective_bits + overhead
        est_ratio = 32 / (effective_bits + 0.5) if effective_bits > 0 else 0
        
        # Create a text report figure
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis('off')
        
        text = f"""Compression Potential Analysis
        
Entropy (H): {ent:.4f} bits
Unique Values: {n_unique}
Unique Ratio: {unique_ratio:.2%}
Effective Bits: {effective_bits:.2f}

Estimated Compression Ratio (vs FP32): {est_ratio:.2f}x
        """
        
        ax.text(0.1, 0.5, text, fontsize=14, family='monospace')
        return fig

    def visualize_3d_structure(self, weights: Dict[str, torch.Tensor], layer_name: Optional[str] = None):
        """7. Interactive 3D Visualization using PCA"""
        # Select layer
        target_w = None
        target_name = ""
        
        if layer_name and layer_name in weights:
            target_w = weights[layer_name]
            target_name = layer_name
        else:
            for name, param in weights.items():
                if param.dim() == 2 and param.shape[0] > 10 and param.shape[1] > 3:
                    target_w = param
                    target_name = name
                    break
        
        if target_w is None:
            return None
            
        # Use PCA to reduce to 3D
        # Treat rows as data points
        from sklearn.decomposition import PCA
        
        w_np = self._tensor_to_numpy(target_w)
        # Downsample rows
        if w_np.shape[0] > 1000:
            w_np = w_np[:1000]
            
        pca = PCA(n_components=3)
        try:
            result = pca.fit_transform(w_np)
            
            df = pd.DataFrame(result, columns=['PC1', 'PC2', 'PC3'])
            df['Index'] = range(len(df))
            
            fig = px.scatter_3d(df, x='PC1', y='PC2', z='PC3', 
                               color='Index', 
                               title=f"3D PCA of {target_name} Rows")
            return fig
        except Exception as e:
            logger.warning(f"PCA failed: {e}")
            return None

    def generate_comprehensive_report(self, weights: Dict[str, torch.Tensor], output_dir: str):
        """Generate all plots and save to directory"""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Dist
        fig = self.visualize_weight_distribution(weights)
        if fig:
            fig.savefig(out_path / "distribution.png")
            plt.close(fig)
            
        # 2. Heatmap
        fig = self.visualize_weight_heatmap(weights)
        if fig:
            fig.savefig(out_path / "heatmap.png")
            plt.close(fig)
            
        # 3. Sparsity
        fig = self.visualize_sparsity_pattern(weights)
        if fig:
            fig.savefig(out_path / "sparsity.png")
            plt.close(fig)
            
        # 4. Freq
        fig = self.visualize_value_frequency(weights)
        if fig:
            fig.savefig(out_path / "frequency.png")
            plt.close(fig)
            
        # 6. Compression
        fig = self.visualize_compression_potential(weights)
        if fig:
            fig.savefig(out_path / "compression.png")
            plt.close(fig)
            
        # 7. 3D (HTML)
        fig_3d = self.visualize_3d_structure(weights)
        if fig_3d:
            fig_3d.write_html(out_path / "structure_3d.html")
            
        logger.info(f"Visualization report saved to {output_dir}")
