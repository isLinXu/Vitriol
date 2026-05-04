from torch import nn
import logging
import os

logger = logging.getLogger(__name__)

class VitriolVisualizer:
    """Visualizes PyTorch model architectures."""
    
    @staticmethod
    def generate_diagram(model: nn.Module, output_path: str, dummy_input_shape=(1, 128)):
        """
        Generates a visualization of the model architecture.
        Uses pure text-based 'repr(model)' rendering to avoid heavy dependencies like graphviz.
        """
        try:
            # We use PIL to draw text to image
            from PIL import Image, ImageDraw, ImageFont
            
            text = str(model)
            # Limit lines to prevent massive images
            lines = text.split('\n')
            
            # Intelligent truncation: keep start and end, truncate repeated layers
            if len(lines) > 100:
                # Find repeating blocks? Simple approach: keep first 40 and last 40 lines
                lines = lines[:40] + [f"... {len(lines)-80} lines truncated ..."] + lines[-40:]
            
            # Formatting
            font_size = 14
            line_height = 18
            padding = 20
            
            # Estimate width based on max line length
            max_line_len = max(len(line) for line in lines)
            width = max(800, max_line_len * 8 + padding * 2)
            height = len(lines) * line_height + padding * 2
            
            # Create high-quality image
            img = Image.new('RGB', (width, height), color='#F5F5F5') # Light gray background
            d = ImageDraw.Draw(img)
            
            # Try to load a monospace font
            try:
                # Common locations for monospace fonts
                font_paths = [
                    "/System/Library/Fonts/Monaco.ttf", # macOS
                    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", # Linux
                    "arial.ttf" # Windows/Fallback
                ]
                font = None
                for path in font_paths:
                    if os.path.exists(path):
                        font = ImageFont.truetype(path, font_size)
                        break
                if not font:
                    font = ImageFont.load_default()
            except OSError:
                font = ImageFont.load_default()

            # Draw text
            y_text = padding
            for line in lines:
                d.text((padding, y_text), line, font=font, fill='#333333')
                y_text += line_height
            
            # Add header
            # d.text((padding, 5), f"Model Architecture: {model.__class__.__name__}", font=font, fill='blue')
            
            img.save(output_path)
            logger.info("Generated textual architecture image at %s", output_path)
            return True
            
        except ImportError:
            logger.warning("PIL not installed. Saving architecture.txt instead.")
            with open(output_path.replace('.png', '.txt'), 'w') as f:
                f.write(str(model))
            return False
        except Exception as e:
            logger.error("Failed to generate textual image: %s", e)
            return False
