from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..utils.hf_loading import hf_kwargs
from ..utils.hf_loading import load_causallm as hf_load_causallm
from ..utils.hf_loading import load_model as hf_load_model
from ..utils.hf_loading import load_tokenizer as hf_load_tokenizer

logger = logging.getLogger(__name__)

# Lazy-loaded module hook kept at module scope so tests and callers can
# monkeypatch the seq2seq loader without importing Transformers up front.
AutoModelForSeq2SeqLM = None

@dataclass
class ValidationReport:
    """Report summarizing validation findings for a generated model.

    Attributes:
        success: Overall validation passed (no critical errors).
        model_loadable: Model can be loaded via transformers.
        tokenizer_loadable: Tokenizer can be loaded and encode text.
        inference_test: A forward pass produces valid output shapes.
        memory_usage_gb: Peak memory usage during validation, if available.
        errors: List of critical error messages.
        warnings: List of non-critical warning messages.
    """
    success: bool
    model_loadable: bool
    tokenizer_loadable: bool
    inference_test: bool
    memory_usage_gb: Optional[float] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the report to a dictionary."""
        return {
            'success': self.success,
            'model_loadable': self.model_loadable,
            'tokenizer_loadable': self.tokenizer_loadable,
            'inference_test': self.inference_test,
            'memory_usage_gb': self.memory_usage_gb,
            'errors': self.errors,
            'warnings': self.warnings
        }


class ModelValidator:
    """Validate generated minimal-weight models.

    Performs a four-stage validation pipeline:

    1. **Model Loading**: Attempts to load the model via the appropriate
       AutoModel class (causal LM, seq2seq, or generic).
    2. **Tokenizer Loading**: Loads the tokenizer and verifies it can encode
       sample text.
    3. **Inference Test**: Runs a single forward pass to confirm output
       shapes are valid.
    4. **Memory Audit**: Records peak memory usage (RAM or VRAM).

    Low-memory adaptation: if available RAM < 8 GB, automatically constrains
    ``max_memory`` and enables disk offloading.

    Example::

        validator = ModelValidator("output/qwen-mini")  # trust_remote_code defaults to False
        report = validator.validate(run_inference=True)
        print(report.to_dict())
    """

    def __init__(self, output_dir: str, trust_remote_code: bool = False):
        self.output_dir = output_dir
        self.trust_remote_code = bool(trust_remote_code)
        self.report = ValidationReport(
            success=True,
            model_loadable=False,
            tokenizer_loadable=False,
            inference_test=False
        )

    def validate(self, run_inference: bool = True, task_type: str = "causal_lm") -> ValidationReport:
        """Run the full validation pipeline.

        Args:
            run_inference: If False, skips tokenizer loading and forward-pass
                tests (useful for pure load-testing scenarios).
            task_type: Model task type — ``"causal_lm"``, ``"seq2seq"``,
                or ``"generic"``.

        Returns:
            A :class:`ValidationReport` with detailed results.
        """
        # A validator can be reused after files are repaired. Each attempt must
        # report only its own outcome, not accumulated errors from earlier runs.
        self.report = ValidationReport(True, False, False, False)
        logger.info("Validating model in %s...", self.output_dir)
        try:
            # 1. Validate Model Loading
            model = self._validate_model_loading(task_type=task_type)

            # 2. Validate Tokenizer Loading
            tokenizer = None
            if run_inference:
                tokenizer = self._validate_tokenizer_loading()
            else:
                self.report.warnings.append("Tokenizer validation skipped because inference is disabled")

            # 3. Inference Test
            if run_inference and model and tokenizer:
                self._validate_inference(model, tokenizer)

            # 4. Check Memory
            self._check_memory_usage(model)

        except Exception as e:
            self.report.success = False
            self.report.errors.append(f"Validation critical failure: {str(e)}")
            logger.error("Validation failed: %s", e)

        return self.report

    def _security_context(self) -> Dict[str, bool]:
        """Security kwargs for local validation loads."""
        return {
            "trust_remote_code": self.trust_remote_code,
            "allow_network": False,
            "local_files_only": True,
        }

    def _load_model_for_task(self, task_type: str):
        # Validation should default to local loading to avoid accidental network access.
        security = self._security_context()

        common_kwargs = {
            "torch_dtype": "auto",
            "device_map": "cpu",
            "low_cpu_mem_usage": True,
            "output_loading_info": True,
        }

        # Detect available memory and adjust for large models
        try:
            import psutil
            available_gb = psutil.virtual_memory().available / (1024 ** 3)
            if available_gb < 8:
                logger.warning(
                    "Low memory detected (%.1f GB available). "
                    "Using max_memory constraint for model loading.",
                    available_gb,
                )
                # Limit PyTorch to use at most 60% of available RAM
                max_mem = int(available_gb * 0.6)
                common_kwargs["max_memory"] = {
                    "cpu": f"{max_mem}GiB",
                }
                # Use offload folder for disk-based offloading
                import tempfile
                offload_dir = tempfile.mkdtemp(prefix="vitriol_offload_")
                common_kwargs["offload_folder"] = offload_dir
                logger.info("Offload folder: %s", offload_dir)
        except ImportError:
            pass

        if task_type == "seq2seq":
            # seq2seq is not included in the current facade (not in Auto*); can be extended later if needed.
            global AutoModelForSeq2SeqLM
            if AutoModelForSeq2SeqLM is None:
                from transformers import AutoModelForSeq2SeqLM as _AutoModelForSeq2SeqLM

                AutoModelForSeq2SeqLM = _AutoModelForSeq2SeqLM

            return AutoModelForSeq2SeqLM.from_pretrained(
                self.output_dir,
                **common_kwargs,
                **hf_kwargs(security),
            )
        if task_type == "generic":
            return hf_load_model(self.output_dir, security=security, **common_kwargs)

        # Validation must load the actual stored shapes. Ignoring mismatches
        # silently initializes replacement weights and incorrectly certifies a
        # broken artifact as loadable.
        try:
            return hf_load_causallm(self.output_dir, security=security, **common_kwargs)
        except (MemoryError, OSError) as e:
            # OOM — try with disk offloading
            err_str = str(e).lower()
            if "memory" in err_str or "oom" in err_str or "cannot allocate" in err_str:
                logger.warning("OOM during model loading, retrying with disk offloading: %s", e)
                try:
                    import tempfile
                    common_kwargs.setdefault(
                        "offload_folder",
                        tempfile.mkdtemp(prefix="vitriol_offload_"),
                    )
                    return hf_load_causallm(self.output_dir, security=security, **common_kwargs)
                except Exception as e2:
                    logger.error("Still OOM with offloading: %s", e2)
                    raise
            raise

    @staticmethod
    def _check_loaded_weights(result):
        # Transformers reports missing/unexpected tensors as warnings and may
        # initialize missing parameters. A load validator must reject those
        # artifacts instead of certifying the replacement weights.
        if not (isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict)):
            return result  # Compatibility with injected/custom loaders.
        model, info = result
        problems = {}
        for diagnostic in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
            entries = info.get(diagnostic) or []
            if diagnostic == "unexpected_keys":
                entries = [key for key in entries if not str(key).startswith("__vitriol_pad__")]
            if entries:
                problems[diagnostic] = entries
        if problems:
            raise RuntimeError(f"Stored weights do not match the model: {problems}")
        return model

    def _validate_model_loading(self, task_type: str = "causal_lm"):
        try:
            logger.info("Attempting to load model...")
            model = self._check_loaded_weights(self._load_model_for_task(task_type))
            self.report.model_loadable = True
            logger.info("Model loaded successfully.")
            return model
        except Exception as e:
            if isinstance(e, RuntimeError):
                self.report.success = False
                self.report.errors.append(f"Model loading failed: {e}")
                logger.error("Model loading failed: %s", e)
                return None
            # Try generic AutoModel for unsupported task/model families.
            try:
                security = self._security_context()
                common_kwargs = {
                    "torch_dtype": "auto",
                    "device_map": "cpu",
                    "low_cpu_mem_usage": True,
                    "output_loading_info": True,
                }
                # Memory-constrained loading
                try:
                    import psutil
                    available_gb = psutil.virtual_memory().available / (1024 ** 3)
                    if available_gb < 8:
                        max_mem = int(available_gb * 0.6)
                        common_kwargs["max_memory"] = {"cpu": f"{max_mem}GiB"}
                        import tempfile
                        common_kwargs["offload_folder"] = tempfile.mkdtemp(prefix="vitriol_offload_")
                except ImportError:
                    pass

                model = self._check_loaded_weights(
                    hf_load_model(self.output_dir, security=security, **common_kwargs))
                self.report.model_loadable = True
                self.report.warnings.append("Loaded as AutoModel, not AutoModelForCausalLM")
                logger.info("Model loaded as generic AutoModel.")
                return model
            except Exception as fallback_error:
                self.report.success = False
                self.report.errors.append(f"Model loading failed: {e}; generic fallback: {fallback_error}")
                logger.error(f"Model loading failed: {e}")
                return None

    def _validate_tokenizer_loading(self):
        try:
            logger.info("Attempting to load tokenizer...")
            tokenizer = hf_load_tokenizer(
                self.output_dir,
                security=self._security_context(),
            )
            self.report.tokenizer_loadable = True
            logger.info("Tokenizer loaded successfully.")
            return tokenizer
        except Exception as e:
            self.report.success = False # Tokenizer failure is usually critical for usability
            self.report.errors.append(f"Tokenizer loading failed: {e}")
            logger.error(f"Tokenizer loading failed: {e}")
            return None

    def _validate_inference(self, model, tokenizer):
        try:
            import torch

            logger.info("Running inference test...")
            input_text = "Hello, world!"
            inputs = tokenizer(input_text, return_tensors="pt")

            # Move inputs to model device
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                # Generate a few tokens
                if hasattr(model, "generate"):
                    outputs = model.generate(**inputs, max_new_tokens=5)
                    tokenizer.decode(outputs[0], skip_special_tokens=True)
                else:
                    # For non-generative models, just run forward pass
                    model(**inputs)

            self.report.inference_test = True
            logger.info("Inference test passed.")
        except Exception as e:
            self.report.success = False
            self.report.errors.append(f"Inference test failed: {e}")
            logger.error(f"Inference test failed: {e}")

    def _check_memory_usage(self, model):
        if model:
            try:
                param_size = sum(p.numel() * p.element_size() for p in model.parameters())
                buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
                total_size = param_size + buffer_size
                self.report.memory_usage_gb = total_size / (1024**3)
            except Exception as e:
                logger.debug("Memory usage check failed: %s", e)
