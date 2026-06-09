
from typing import Any

import matplotlib.patches as patches
import matplotlib.pyplot as plt

from ..core import Architecture


class BlockRenderer:
    """Renders a high-level block diagram of the architecture."""

    def __init__(self, style: str = 'default'):
        self.style = style
        # Professional color palette (Academic/Modern)
        if style == 'academic':
            self.colors = {
                'embedding': '#F0F0F0', 'attention': '#E0E0E0', 'feedforward': '#D0D0D0',
                'normalization': '#F5F5F5', 'output': '#E5E5E5', 'block': '#FFFFFF'
            }
            self.edge_color = '#000000'
            self.text_color = '#000000'
        else:
            # Modern pastel colors
            self.colors = {
                'embedding': '#B3E5FC', # Light Blue
                'attention': '#FFE0B2', # Light Orange
                'feedforward': '#C8E6C9', # Light Green
                'feedforward_dense': '#C8E6C9',
                'feedforward_moe': '#E1BEE7',
                'normalization': '#F5F5F5', # Light Grey
                'output': '#FFCDD2', # Light Red
                'adapter': '#D1C4E9', # Light Purple
                'block': '#FAFAFA'
            }
            self.edge_color = '#E0E0E0'
            self.text_color = '#333333'

    def render(self, architecture: Architecture, output_path: str) -> Any:
        fig, ax = plt.subplots(figsize=(10, 16))
        ax.set_axis_off()

        # Drawing constants
        box_width = 1.2
        box_height = 0.6
        gap = 0.4
        x_center = 0.5
        y_pos = 14.0

        # Title
        title_color = '#1A1A1A'
        ax.text(x_center, y_pos, f"{architecture.model_type.upper()}",
                ha='center', fontsize=24, weight='bold', color=title_color, family='sans-serif')
        ax.text(x_center, y_pos - 0.6, f"{architecture.total_params/1e9:.2f}B Parameters | {architecture.total_layers} Layers",
                ha='center', fontsize=14, color='#666666', family='sans-serif')
        self._draw_feature_badges(ax, architecture, x_center, y_pos - 1.15)
        y_pos -= 2.0

        # Helper to decide text color based on background
        def get_text_color(bg_color) -> Any:
            return self.text_color

        # Draw Input
        self._draw_box(ax, x_center, y_pos, box_width, box_height, "Input Tokens", "#FFFFFF", '#000000', '#E0E0E0')
        y_pos -= (box_height + gap)

        # Draw Embedding
        emb_layer = next((layer for layer in architecture.layers if layer.type == 'embedding'), None)
        if emb_layer:
            self._draw_box(ax, x_center, y_pos, box_width, box_height,
                           f"Token Embedding\n{emb_layer.shape}", self.colors['embedding'],
                           self.text_color, self.edge_color)
            y_pos -= (box_height + gap)

        # Draw Transformer Block Container
        y_pos -= 0.5
        block_top = y_pos + 0.5

        # Norm 1
        self._draw_box(ax, x_center, y_pos, box_width*0.8, box_height, "RMSNorm",
                       self.colors['normalization'], self.text_color, self.edge_color)

        # Residual connection 1 (curve around Norm + Attn)
        # Start: above Norm 1, End: below Attn
        res_start_y = y_pos + box_height/2 + gap/2

        y_pos -= (box_height + gap)

        # Attention
        attn_info = next((layer.description for layer in architecture.layers if layer.type == 'attention'), "")
        # GQA Visualization text
        if "GQA" in architecture.special_features:
             attn_info = f"GQA Attention\n{attn_info}"

        self._draw_box(ax, x_center, y_pos, box_width, box_height, f"{attn_info}",
                       self.colors['attention'], self.text_color, self.edge_color)

        res_end_y = y_pos - box_height/2 - gap/2
        self._draw_residual(ax, res_start_y, res_end_y, box_width)

        y_pos -= (box_height + gap)

        # Norm 2
        res_start_y_2 = y_pos + box_height/2 + gap/2
        self._draw_box(ax, x_center, y_pos, box_width*0.8, box_height, "RMSNorm",
                       self.colors['normalization'], self.text_color, self.edge_color)
        y_pos -= (box_height + gap)

        # FFN
        # Shape as width metaphor (trapezoid?) - stick to box for simplicity but wider
        ffn_info = next((layer.description for layer in architecture.layers if layer.type == 'feedforward'), "")
        ffn_color = self._feedforward_color(architecture)
        self._draw_box(ax, x_center, y_pos, box_width*1.1, box_height, f"Feed Forward (SwiGLU)\n{ffn_info}",
                       ffn_color, self.text_color, self.edge_color)

        res_end_y_2 = y_pos - box_height/2 - gap/2
        self._draw_residual(ax, res_start_y_2, res_end_y_2, box_width*1.1)

        y_pos -= (box_height + gap)

        block_bottom = y_pos + 0.3

        # Draw surrounding box for block
        rect = patches.FancyBboxPatch((x_center - box_width/2 - 0.4, block_bottom),
                                 box_width + 0.8, block_top - block_bottom,
                                 boxstyle="round,pad=0.1",
                                 linewidth=1.5, edgecolor='#999999', facecolor='none', linestyle='--', zorder=0)
        ax.add_patch(rect)

        # Label for block repetition
        ax.text(x_center + box_width/2 + 0.5, (block_top + block_bottom)/2,
                f"× {architecture.parameters.get('num_layers', 'N')}",
                ha='left', va='center', fontsize=18, weight='bold', color='#333333')
        self._draw_hy3_legend(ax, architecture, x_center + box_width/2 + 0.7, block_top - 0.1)

        y_pos -= 0.8

        # Final Norm
        self._draw_box(ax, x_center, y_pos, box_width*0.8, box_height, "Final RMSNorm",
                       self.colors['normalization'], self.text_color, self.edge_color)
        y_pos -= (box_height + gap)

        mtp_layers = int(architecture.parameters.get("mtp_layers", 0) or 0)
        if mtp_layers > 0:
            self._draw_box(
                ax,
                x_center,
                y_pos,
                box_width,
                box_height,
                f"MTP Head\nNext-N layers: {mtp_layers}",
                self.colors.get('adapter', '#D1C4E9'),
                self.text_color,
                self.edge_color,
            )
            y_pos -= (box_height + gap)

        # Output Head
        self._draw_box(ax, x_center, y_pos, box_width, box_height, "LM Head",
                       self.colors['output'], self.text_color, self.edge_color)

        # Adjust limits
        ax.set_xlim(-0.5, 2.0)
        ax.set_ylim(y_pos - 1, 15)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    def _feedforward_color(self, architecture: Architecture) -> str:
        if architecture.model_type == "hy_v3" and "MoE" in architecture.special_features:
            return self.colors.get("feedforward_moe", self.colors["feedforward"])
        return self.colors.get("feedforward_dense", self.colors["feedforward"])

    def _draw_feature_badges(self, ax, architecture: Architecture, x_center: float, y: float) -> None:
        labels = []
        if architecture.model_type == "hy_v3":
            dense_prefix = int(architecture.parameters.get("dense_prefix_layers", 0) or 0)
            num_experts = int(architecture.parameters.get("num_experts", 0) or 0)
            top_k = int(architecture.parameters.get("top_k_experts", 0) or 0)
            mtp_layers = int(architecture.parameters.get("mtp_layers", 0) or 0)
            if dense_prefix > 0:
                labels.append(("Dense Prefix", "#A5D6A7"))
            if num_experts > 0 and top_k > 0:
                labels.append((f"MoE top-{top_k}/{num_experts}", "#CE93D8"))
            if "GQA" in architecture.special_features:
                labels.append(("GQA", "#FFCC80"))
            if mtp_layers > 0:
                labels.append((f"MTP x{mtp_layers}", "#B39DDB"))
        if not labels:
            return
        x = x_center - 0.9
        for text, color in labels:
            badge = patches.FancyBboxPatch(
                (x, y - 0.16),
                0.42,
                0.18,
                boxstyle="round,pad=0.03,rounding_size=0.04",
                linewidth=0.8,
                edgecolor=self.edge_color,
                facecolor=color,
                zorder=1,
            )
            ax.add_patch(badge)
            ax.text(x + 0.21, y - 0.07, text, ha='center', va='center', fontsize=8.5, color='#333333', zorder=2)
            x += 0.48

    def _draw_hy3_legend(self, ax, architecture: Architecture, x: float, y: float) -> None:
        if architecture.model_type != "hy_v3":
            return
        dense_prefix = int(architecture.parameters.get("dense_prefix_layers", 0) or 0)
        total_layers = int(architecture.parameters.get("num_layers", architecture.parameters.get("total_layers", 0)) or 0)
        moe_layers = max(total_layers - dense_prefix, 0)
        legend_items = [
            ("Dense prefix", self.colors.get("feedforward_dense", "#C8E6C9"), f"× {dense_prefix}"),
            ("MoE blocks", self.colors.get("feedforward_moe", "#E1BEE7"), f"× {moe_layers}"),
        ]
        mtp_layers = int(architecture.parameters.get("mtp_layers", 0) or 0)
        if mtp_layers > 0:
            legend_items.append(("MTP head", self.colors.get("adapter", "#D1C4E9"), f"× {mtp_layers}"))

        for idx, (label, color, value) in enumerate(legend_items):
            yy = y - idx * 0.35
            rect = patches.FancyBboxPatch(
                (x, yy - 0.12),
                0.28,
                0.16,
                boxstyle="round,pad=0.02,rounding_size=0.03",
                linewidth=0.8,
                edgecolor=self.edge_color,
                facecolor=color,
                zorder=1,
            )
            ax.add_patch(rect)
            ax.text(x + 0.34, yy - 0.04, f"{label} {value}", ha='left', va='center', fontsize=9, color='#444444')

    def _draw_box(self, ax, x, y, w, h, text, color, text_color, edge_color):
        # Drop shadow effect
        shadow = patches.FancyBboxPatch((x - w/2 + 0.02, y - h/2 - 0.02), w, h,
                                   boxstyle="round,pad=0.05,rounding_size=0.05",
                                   linewidth=0,
                                   facecolor='#000000',
                                   alpha=0.1,
                                   zorder=0)
        ax.add_patch(shadow)

        box = patches.FancyBboxPatch((x - w/2, y - h/2), w, h,
                                   boxstyle="round,pad=0.05,rounding_size=0.05",
                                   linewidth=1 if edge_color != 'none' else 0,
                                   edgecolor=edge_color,
                                   facecolor=color,
                                   zorder=1)
        ax.add_patch(box)

        # Split text for better formatting
        lines = text.split('\n')
        main_text = lines[0]
        sub_text = '\n'.join(lines[1:]) if len(lines) > 1 else ""

        ax.text(x, y + (0.05 if sub_text else 0), main_text, ha='center', va='center', fontsize=12, color=text_color, zorder=2, weight='bold', family='sans-serif')
        if sub_text:
            ax.text(x, y - 0.15, sub_text, ha='center', va='center', fontsize=10, color='#666666', zorder=2, family='monospace')

    def _draw_residual(self, ax, start_y, end_y, width):
        # Draw curved line from start_y to end_y around the right side
        # Control points for curve
        x_offset = width/2 + 0.1

        [0.5, 0.5 + x_offset, 0.5 + x_offset, 0.5]

        # Using Bezier curve via PathPatch is complex, let's use simple plot with smoothing
        # Or just rect connection for now to be clean

        # Line out
        ax.plot([0.5, 0.5 + x_offset], [start_y, start_y], color='#999999', linewidth=1.5, zorder=0)
        # Vertical line
        ax.plot([0.5 + x_offset, 0.5 + x_offset], [start_y, end_y], color='#999999', linewidth=1.5, zorder=0)
        # Line in
        ax.plot([0.5 + x_offset, 0.5], [end_y, end_y], color='#999999', linewidth=1.5, zorder=0)

        # Add label
        ax.text(0.5 + x_offset + 0.05, (start_y + end_y)/2, "Residual", ha='left', va='center', fontsize=9, color='#999999', rotation=90)

        # Add "+" circle at merge point
        circle = patches.Circle((0.5, end_y), 0.08, facecolor='#FFFFFF', edgecolor='#999999', linewidth=1.5, zorder=2)
        ax.add_patch(circle)
        ax.text(0.5, end_y, "+", ha='center', va='center', fontsize=12, color='#999999', weight='bold', zorder=3)
