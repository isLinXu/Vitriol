
from .base import ModelAdapter as ModelAdapter
from .registry import AdapterRegistry as AdapterRegistry

# Concrete adapter classes — lazy-import to avoid forcing heavy deps at import time.
# Users can: from vitriol.adapters import LlamaAdapter
from .llama import LlamaAdapter as LlamaAdapter
from .qwen import QwenMoeAdapter as QwenMoeAdapter, Qwen35MoeAdapter as Qwen35MoeAdapter
from .deepseek import DeepSeekAdapter as DeepSeekAdapter
from .mistral import MistralAdapter as MistralAdapter
from .gemma import GemmaAdapter as GemmaAdapter
from .phi import PhiAdapter as PhiAdapter
from .cohere import CohereAdapter as CohereAdapter
from .glm import GLMAdapter as GLMAdapter
from .stablelm import StableLMAdapter as StableLMAdapter
from .minimax import MiniMaxAdapter as MiniMaxAdapter
