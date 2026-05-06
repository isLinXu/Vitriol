"""Architecture renderers — block, detail and HTML output."""

from .block import BlockRenderer
from .detail import DetailRenderer
from .html import HTMLRenderer

__all__ = [
    "BlockRenderer",
    "DetailRenderer",
    "HTMLRenderer",
]
