
import matplotlib.pyplot as plt
import pandas as pd

from ..core import Architecture


class DetailRenderer:
    """Renders a detailed list of layers using pandas styling."""

    def render(self, architecture: Architecture, output_path: str) -> None:
        # Prepare data
        data = []
        for layer in architecture.layers:
            if layer.type in ['block_start', 'block_end']:
                continue
            data.append({
                "Layer Name": layer.name,
                "Role": self._role_for_layer(architecture, layer),
                "Type": layer.type.upper(),
                "Shape": str(layer.shape),
                "Params": f"{layer.params:,}",
                "Description": layer.description
            })

        df = pd.DataFrame(data)

        # Limit rows for static image to avoid huge files
        max_rows = 50
        if len(df) > max_rows:
            # Keep first 20 and last 20, insert ellipses
            df_head = df.iloc[:20]
            df_tail = df.iloc[-20:]
            df_mid = pd.DataFrame([{"Layer Name": "...", "Role": "...", "Type": "...", "Shape": "...", "Params": "...", "Description": "..."}])
            df = pd.concat([df_head, df_mid, df_tail])

        # Render as table using matplotlib
        # Increase width and adjust height
        fig, ax = plt.subplots(figsize=(18, len(df) * 0.5 + 3))
        ax.set_axis_off()

        # Title
        ax.text(0.5, 0.98, f"Detailed Architecture: {architecture.model_type.upper()}",
                ha='center', fontsize=24, weight='bold', transform=ax.transAxes, family='sans-serif', color='#1A1A1A')

        # Table
        table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='left',
                        colWidths=[0.25, 0.12, 0.13, 0.17, 0.13, 0.2])

        # Style
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.0) # More vertical padding

        # Header styling
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor('#E0E0E0')
            cell.set_linewidth(1)

            if row == 0:
                cell.set_text_props(weight='bold', color='#333333', family='sans-serif')
                cell.set_facecolor('#F5F5F5')
                cell.set_height(0.06)
            else:
                cell.set_text_props(family='monospace', color='#1A1A1A')
                cell.set_height(0.05)
                # Alternate row colors
                if row % 2 == 0:
                    cell.set_facecolor('#FAFAFA')
                else:
                    cell.set_facecolor('#FFFFFF')
                role = df.iloc[row - 1]["Role"] if row - 1 < len(df) else ""
                if role == "Dense Prefix":
                    cell.set_facecolor('#F1F8E9')
                elif role == "MoE Block":
                    cell.set_facecolor('#F3E5F5')
                elif role == "MTP":
                    cell.set_facecolor('#EDE7F6')

                # Right align Params column (index 4)
                if col == 4:
                    cell.get_text().set_horizontalalignment('right')
                    cell.set_text_props(weight='bold', family='monospace')

                # Type column styling
                if col in (1, 2):
                    cell.set_text_props(weight='bold', size=9)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.close(fig)

    @staticmethod
    def _role_for_layer(architecture: Architecture, layer) -> str:
        if architecture.model_type != "hy_v3":
            return ""
        name = str(layer.name)
        description = str(layer.description)
        if "MTP" in name or "Next-N" in description:
            return "MTP"
        if "Block 0" in name and "Dense" in description:
            return "Dense Prefix"
        if "MoE" in description or "TopK:" in description:
            return "MoE Block"
        return ""
