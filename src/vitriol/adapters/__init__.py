
from .base import ModelAdapter as ModelAdapter
from .cohere import CohereAdapter as CohereAdapter
from .deepseek import DeepSeekAdapter as DeepSeekAdapter
from .gemma import GemmaAdapter as GemmaAdapter
from .glm import GLMAdapter as GLMAdapter

# Concrete adapter classes — lazy-import to avoid forcing heavy deps at import time.
# Users can: from vitriol.adapters import LlamaAdapter
from .llama import LlamaAdapter as LlamaAdapter
from .minimax import MiniMaxAdapter as MiniMaxAdapter
from .mistral import MistralAdapter as MistralAdapter
from .phi import PhiAdapter as PhiAdapter
from .qwen import Qwen35MoeAdapter as Qwen35MoeAdapter
from .qwen import QwenMoeAdapter as QwenMoeAdapter
from .registry import AdapterRegistry as AdapterRegistry
from .stablelm import StableLMAdapter as StableLMAdapter

__all__ = [
    "ModelAdapter",
    "AdapterRegistry",
    "CohereAdapter",
    "DeepSeekAdapter",
    "GemmaAdapter",
    "GLMAdapter",
    "LlamaAdapter",
    "MiniMaxAdapter",
    "MistralAdapter",
    "PhiAdapter",
    "QwenMoeAdapter",
    "Qwen35MoeAdapter",
    "StableLMAdapter",
]
