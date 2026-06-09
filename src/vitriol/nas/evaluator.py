import json
import logging
import math
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from vitriol.config.manager import GenerationConfig
from vitriol.core.generator import MinimalWeightGenerator

from .search_space import ArchitectureGene

logger = logging.getLogger(__name__)

class ZeroCostProxy:
    """Base class for zero-cost proxies."""
    def score(self, gene: ArchitectureGene, model: torch.nn.Module = None, inputs: torch.Tensor = None, targets: torch.Tensor = None) -> float:
        raise NotImplementedError

class ParamCountProxy(ZeroCostProxy):
    """Proxy that uses parameter count (or negative parameter count) as a score."""
    def score(self, gene: ArchitectureGene, model: torch.nn.Module = None, inputs: torch.Tensor = None, targets: torch.Tensor = None) -> float:
        V = getattr(gene, 'vocab_size', 32000)
        H = gene.hidden_size
        N = gene.n_layers
        I_size = gene.intermediate_size

        emb = V * H
        attn = 4 * H * H
        mlp = 3 * H * I_size
        layer = attn + mlp + 2 * H # + norms

        total = emb + N * layer + H * V # output head
        return float(total)

class GradNormProxy(ZeroCostProxy):
    """
    Gradient Norm Proxy.
    Calculates the sum of Euclidean norms of gradients for all parameters.
    Higher gradient norm usually implies better trainability (vanishing gradient problem check).
    """
    def score(self, gene: ArchitectureGene, model: torch.nn.Module = None, inputs: torch.Tensor = None, targets: torch.Tensor = None) -> float:
        if model is None or inputs is None or targets is None:
             raise ValueError("GradNormProxy requires model, inputs and targets")

        model.zero_grad()
        outputs = model(inputs, labels=targets)
        loss = outputs.loss
        loss.backward()

        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.detach().data.norm(2).item() ** 2

        return grad_norm ** 0.5

class SynflowProxy(ZeroCostProxy):
    """
    Synaptic Flow Proxy (Synflow).
    Computes the product of weights and their gradients (R_syn = w * dL/dw).
    This measures the 'flow' of information through the network.
    Ideally, we sum |w * grad| over all parameters.
    """
    def score(self, gene: ArchitectureGene, model: torch.nn.Module = None, inputs: torch.Tensor = None, targets: torch.Tensor = None) -> float:
        if model is None or inputs is None:
             raise ValueError("SynflowProxy requires model and inputs")

        # Synflow requires a specific backward pass:
        # 1. Convert all params to positive
        # 2. Forward with all ones input
        # 3. Sum output as loss
        # 4. Backward
        # 5. Score = sum(|w * grad|)

        # Save original weights and state
        orig_weights = {}
        try:
            for name, p in model.named_parameters():
                orig_weights[name] = p.data.clone()
                p.data.abs_() # Make weights positive

            model.zero_grad()

            # Override input to be all ones (or standard input, but Synflow paper suggests specific input)
            # For LLM, we can just use standard input but sum the output logits
            # Or just use the loss from standard input as a proxy for "output magnitude"

            # Simplified implementation for LLM context:
            outputs = model(inputs)
            # Sum of all logits/loss to create gradients for all paths
            loss = outputs.logits.sum()
            loss.backward()

            score = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    score += (p.data * p.grad).abs().sum().item()

            return score
        finally:
            # Restore weights
            for name, p in model.named_parameters():
                if name in orig_weights:
                    p.data = orig_weights[name]

class FisherProxy(ZeroCostProxy):
    """
    Fisher Proxy.
    Computes the Fisher Information matrix diagonal (squared gradients).
    Measures how much information the parameters contain about the data distribution.
    Score = sum(grad^2)
    """
    def score(self, gene: ArchitectureGene, model: torch.nn.Module = None, inputs: torch.Tensor = None, targets: torch.Tensor = None) -> float:
        if model is None or inputs is None or targets is None:
             raise ValueError("FisherProxy requires model, inputs and targets")

        model.zero_grad()
        outputs = model(inputs, labels=targets)
        loss = outputs.loss
        loss.backward()

        score = 0.0
        for p in model.parameters():
            if p.grad is not None:
                score += (p.grad.detach() ** 2).sum().item()

        return score

class SNIPProxy(ZeroCostProxy):
    """
    SNIP (Single-shot Network Pruning) Proxy.
    Measures connection sensitivity by multiplying weights with their gradients.
    Score = sum(|w * grad|) computed on standard loss (not all-ones input like Synflow).
    """
    def score(self, gene: ArchitectureGene, model: torch.nn.Module = None, inputs: torch.Tensor = None, targets: torch.Tensor = None) -> float:
        if model is None or inputs is None or targets is None:
             raise ValueError("SNIPProxy requires model, inputs and targets")

        model.zero_grad()
        outputs = model(inputs, labels=targets)
        loss = outputs.loss
        loss.backward()

        score = 0.0
        for p in model.parameters():
            if p.grad is not None:
                score += (p.data * p.grad).abs().sum().item()

        return score

class JacobianCovarianceProxy(ZeroCostProxy):
    """
    Jacobian Covariance / NWOT (Neural Networks Without Training) Proxy.
    Evaluates the correlation between activations of different samples.
    Networks that map different inputs to distinct representations score higher.
    Calculates log(det(K)) where K is the correlation matrix of the outputs/logits.
    """
    def score(self, gene: ArchitectureGene, model: torch.nn.Module = None, inputs: torch.Tensor = None, targets: torch.Tensor = None) -> float:
        if model is None or inputs is None:
             raise ValueError("JacobianCovarianceProxy requires model and inputs")

        model.zero_grad()
        with torch.no_grad():
            outputs = model(inputs)

            # Use logits as the representation. Shape: (batch_size, seq_len, vocab_size)
            # Flatten across seq_len and vocab_size to get (batch_size, feature_dim)
            reps = outputs.logits.view(inputs.size(0), -1)

            # Normalize representations
            reps = reps / (reps.norm(dim=1, keepdim=True) + 1e-8)

            # Compute correlation matrix K = X * X^T (batch_size x batch_size)
            K = torch.matmul(reps, reps.t())

            # Add small epsilon to diagonal for numerical stability before logdet
            K = K + torch.eye(K.size(0), device=K.device) * 1e-5

            # Score is log|det(K)| or the sum of log of eigenvalues
            try:
                sign, logabsdet = torch.linalg.slogdet(K)
                score = logabsdet.item()
            except Exception:
                # Fallback to trace if matrix is singular
                score = K.trace().item()

        return score

class RankMeProxy(ZeroCostProxy):
    """
    RankMe / sRank Proxy (2025-2026 standard for LLMs).
    Measures the Effective Rank of the hidden representations.
    A higher effective rank means the architecture is less prone to dimensional collapse
    and has a higher expressivity space at initialization.
    """
    def score(self, gene: ArchitectureGene, model: torch.nn.Module = None, inputs: torch.Tensor = None, targets: torch.Tensor = None) -> float:
        if model is None or inputs is None:
             raise ValueError("RankMeProxy requires model and inputs")

        model.zero_grad()
        with torch.no_grad():
            # Get representations from the model
            outputs = model(inputs, output_hidden_states=True)
            # Use the last hidden state (before LM head)
            hidden_states = outputs.hidden_states[-1] # shape: (batch, seq_len, hidden_dim)

            # Flatten to (batch * seq_len, hidden_dim)
            Z = hidden_states.view(-1, hidden_states.size(-1))

            # Center the representations
            Z = Z - Z.mean(dim=0, keepdim=True)

            # Compute singular values
            # Use float32 for SVD stability
            _, S, _ = torch.linalg.svd(Z.to(torch.float32), full_matrices=False)

            # Compute normalized singular values (probabilities)
            p = (S / S.sum()) + 1e-9

            # Compute Shannon Entropy of singular values
            entropy = -torch.sum(p * torch.log(p))

            # Effective rank is exp(entropy)
            effective_rank = torch.exp(entropy).item()

        return effective_rank

class VitriolExpressivityProxy(ZeroCostProxy):
    """
    Proprietary Zero-cost proxy designed specifically for Vitriol (Next-Gen 2026+).
    Tackles the two deadliest sins of LLM initialization: Over-smoothing and Mode Collapse.

    Combines two novel forward-only metrics:
    1. Dirichlet Energy of Hidden States: Measures resistance to over-smoothing.
       If tokens become indistinguishable in deep layers, energy approaches 0 (collapse).
    2. Vocabulary Entropy: Measures output space utilization.
       If the model collapses to predicting a single token, entropy is 0.
    """
    def score(self, gene: ArchitectureGene, model: torch.nn.Module = None, inputs: torch.Tensor = None, targets: torch.Tensor = None) -> float:
        if model is None or inputs is None:
             raise ValueError("VitriolExpressivityProxy requires model and inputs")

        model.zero_grad()
        with torch.no_grad():
            outputs = model(inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1] # Shape: (Batch, SeqLen, HiddenDim)
            logits = outputs.logits                   # Shape: (Batch, SeqLen, VocabSize)

            # 1. Calculate Dirichlet Energy (Token distinguishability)
            # Normalize hidden states to make it scale-invariant
            h_norm = torch.nn.functional.normalize(hidden_states, p=2, dim=-1)
            # Differences between adjacent tokens in the sequence
            diffs = h_norm[:, 1:, :] - h_norm[:, :-1, :]
            # Mean energy across batch and sequence
            dirichlet_energy = (diffs ** 2).sum(dim=-1).mean().item()

            # Calculate Vocabulary Entropy (Output distribution health)
            # Convert logits to probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)
            # Shannon entropy: -sum(p * log(p))
            entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1).mean().item()

            # Normalize entropy
            vocab_size = logits.size(-1)
            max_entropy = math.log(vocab_size) if vocab_size > 0 else 1.0
            normalized_entropy = entropy / max_entropy

            # The Vitriol Score: Synergy of structural token diversity and semantic output diversity
            # Multiplicative because if either collapses (0), the architecture is fundamentally broken.
            score = dirichlet_energy * normalized_entropy * 100.0

        return score

class VitriolAttentionDiversityProxy(ZeroCostProxy):
    """
    Measures the diversity of attention patterns across different heads.
    If all heads attend to the same tokens, the multi-head mechanism is redundant (Head Collapse).
    Calculates the inverse of the average off-diagonal cosine similarity between attention matrices.
    """
    def score(self, gene: ArchitectureGene, model: torch.nn.Module = None, inputs: torch.Tensor = None, targets: torch.Tensor = None) -> float:
        if model is None or inputs is None:
            raise ValueError("Requires model and inputs")

        model.zero_grad()
        with torch.no_grad():
            try:
                outputs = model(inputs, output_attentions=True)
                attentions = outputs.attentions # Tuple of (Batch, Heads, SeqLen, SeqLen)
                if not attentions:
                    return 0.0

                diversity_score = 0.0
                valid_layers = 0

                for attn in attentions:
                    # attn shape: (B, H, S, S)
                    H = attn.size(1)
                    if H <= 1:
                        continue

                    # Flatten spatial dimensions: (B, H, S*S)
                    attn_flat = attn.view(attn.size(0), H, -1)
                    # Normalize
                    attn_flat = torch.nn.functional.normalize(attn_flat, p=2, dim=-1)

                    # Compute Head-to-Head cosine similarity matrix: (B, H, H)
                    sim_matrix = torch.bmm(attn_flat, attn_flat.transpose(1, 2))

                    # Mask out diagonal
                    mask = torch.eye(H, device=sim_matrix.device).bool()
                    # Average off-diagonal similarity
                    off_diag_sim = sim_matrix[:, ~mask].mean()

                    # Diversity is inversely proportional to similarity
                    layer_diversity = 1.0 - off_diag_sim.item()
                    diversity_score += layer_diversity
                    valid_layers += 1

                return (diversity_score / valid_layers) * 100.0 if valid_layers > 0 else 0.0
            except Exception as e:
                logger.debug(f"Attention diversity proxy failed (likely model doesn't support output_attentions): {e}")
                return 0.0

class HybridEvaluator:
    """Evaluates architectures using a multi-stage process."""

    def __init__(self, output_dir: str, device: str = "cpu", trust_remote_code: bool = False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.param_proxy = ParamCountProxy()
        self.grad_norm_proxy = GradNormProxy()
        self.synflow_proxy = SynflowProxy()
        self.fisher_proxy = FisherProxy()
        self.snip_proxy = SNIPProxy()
        self.nwot_proxy = JacobianCovarianceProxy()
        self.rankme_proxy = RankMeProxy()
        self.vitriol_proxy = VitriolExpressivityProxy()
        self.attn_div_proxy = VitriolAttentionDiversityProxy()

        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Fallback to CPU.")
            self.device = "cpu"
        elif device == "mps" and not torch.backends.mps.is_available():
            logger.warning("MPS requested but not available. Fallback to CPU.")
            self.device = "cpu"
        else:
            self.device = device

        self.trust_remote_code = trust_remote_code

        self.dataset_cache = {}

    def load_dataset(self, dataset_name: str, config_name: str = None, split: str = "train", n_samples: int = 100) -> Optional[Any]:
        """Load and cache a dataset."""
        key = f"{dataset_name}_{config_name}_{split}_{n_samples}"
        if key in self.dataset_cache:
            return self.dataset_cache[key]

        try:
            # Handle local parquet/json files
            if Path(dataset_name).exists():
                logger.info(f"Loading local dataset from {dataset_name}...")
                import datasets
                # Recursively find parquet or json files
                data_files = list(Path(dataset_name).glob("**/*.parquet"))
                if not data_files:
                     data_files = list(Path(dataset_name).glob("**/*.json"))

                if not data_files:
                    logger.warning(f"No parquet/json files found in {dataset_name}")
                    return None

                data_files = [str(f) for f in data_files]
                logger.info(f"Found {len(data_files)} data files.")

                # Use local path
                # Check file type
                # Handle single file or directory logic better

                # Check file type from the first file found
                if data_files[0].endswith(".parquet"):
                    load_type = "parquet"
                elif data_files[0].endswith(".json") or data_files[0].endswith(".jsonl"):
                    load_type = "json"
                else:
                    logger.warning(f"Unknown file type: {data_files[0]}")
                    return None

                dataset = datasets.load_dataset(load_type, data_files=data_files, split="train")

                n_avail = len(dataset)
                n_take = min(n_samples, n_avail)

                # Shuffle and take
                dataset = dataset.shuffle(seed=42).select(range(n_take))

                # Convert to list of dicts
                samples = list(dataset)

                # Verify we actually have data
                if not samples:
                    logger.warning(f"Local dataset {dataset_name} is empty after loading/selection.")
                    return None

                self.dataset_cache[key] = samples
                return samples

            # Handle ModelScope datasets
            elif dataset_name.startswith("ms://") or config_name == "modelscope":
                try:
                    from modelscope.msdatasets import MsDataset
                    # Remove prefix if present
                    ms_name = dataset_name.replace("ms://", "")
                    logger.info(f"Loading ModelScope dataset {ms_name}...")

                    # MsDataset loading
                    # Handle split mapping
                    ms_split = split
                    if split == "train":
                        ms_split = "train"
                    elif split == "validation":
                        ms_split = "validation"
                    elif split == "test":
                        ms_split = "test"

                    ds = MsDataset.load(
                        ms_name,
                        subset_name=config_name if config_name != "modelscope" else None,
                        split=ms_split
                    )

                    # Convert to list of dicts
                    samples = []
                    # MsDataset usually supports iteration
                    count = 0
                    for item in ds:
                        if count >= n_samples:
                            break
                        samples.append(item)
                        count += 1

                    self.dataset_cache[key] = samples
                    return samples
                except ImportError as e:
                    if "No module named 'modelscope'" in str(e):
                        logger.warning("modelscope library not installed. Please install it with `pip install modelscope`.")
                    else:
                        logger.error(f"ModelScope import failed: {e}")
                        import traceback
                        traceback.print_exc()
                    return None
                except Exception as e:
                    logger.error(f"ModelScope load failed: {e}")
                    return None

            else:
                from datasets import load_dataset
                logger.info(f"Loading dataset {dataset_name}...")
                dataset = load_dataset(dataset_name, config_name, split=split, streaming=True)
                samples = list(dataset.take(n_samples))
                self.dataset_cache[key] = samples
                return samples

        except ImportError:
            logger.warning("datasets library not installed. Using dummy data.")
            return None
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return None

    def evaluate(self, gene: ArchitectureGene, strategy: str = "compact", dataset_config: Dict = None) -> Dict[str, Any]:
        """
        Full evaluation pipeline.

        Args:
            dataset_config: Dict with keys 'name', 'config', 'split', 'n_samples'
        """

        # 1. Zero Cost Proxy (Param Count)
        params = self.param_proxy.score(gene)

        loss = 100.0
        flops = 0.0
        grad_norm = 0.0
        synflow = 0.0

        # 2. InMemory Evaluation (No Disk I/O)
        try:
            # Data Preparation & Tokenizer (Load BEFORE model to fix vocab_size)
            input_ids = None
            labels = None
            tokenizer = None

            if dataset_config and dataset_config.get("name"):
                # Use Real Data
                raw_data = self.load_dataset(
                    dataset_config["name"],
                    dataset_config.get("config"),
                    dataset_config.get("split", "train"),
                    dataset_config.get("n_samples", 1024)  # [P1 Fix] Increase samples
                )

                if raw_data:
                    try:
                        # Use Qwen/Qwen2.5-0.5B as default reference if not specified
                        tokenizer_name = dataset_config.get("tokenizer", "Qwen/Qwen2.5-0.5B")
                        from ..utils.hf_loading import load_tokenizer as hf_load_tokenizer

                        tokenizer = hf_load_tokenizer(
                            tokenizer_name,
                            security={
                                "trust_remote_code": self.trust_remote_code,
                                "allow_network": True,
                                "local_files_only": False,
                            },
                        )
                        if tokenizer.pad_token is None:
                            tokenizer.pad_token = tokenizer.eos_token

                        # [P0 Fix] Inject tokenizer vocab size into gene
                        real_vocab = getattr(tokenizer, "vocab_size", 32000)
                        if gene.vocab_size != real_vocab:
                            # logger.info(f"Adjusting gene vocab_size {gene.vocab_size} -> {real_vocab}")
                            gene.vocab_size = real_vocab

                        # Process batch
                        texts = []
                        for x in raw_data:
                            if isinstance(x, dict):
                                if 'text' in x:
                                    texts.append(x['text'])
                                elif 'content' in x:
                                    texts.append(x['content'])
                                else:
                                    texts.append(str(list(x.values())[0]))
                            else:
                                texts.append(str(x))

                        valid_texts = [t for t in texts if t and t.strip()]

                        if valid_texts:
                            encodings = tokenizer(valid_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
                            input_ids = encodings.input_ids.to(self.device)
                            labels = input_ids.clone()

                    except Exception as e:
                        logger.warning(f"Tokenizer load/process failed: {e}. Falling back to dummy data.")
                        input_ids = None

            # Create config from gene (now with correct vocab_size)
            config_dict = gene.to_config()

            from transformers import AutoConfig
            try:
                config = AutoConfig.for_model(**config_dict)
            except Exception:
                config = AutoConfig.from_dict(config_dict)

            # Calculate FLOPs
            flops = self._estimate_flops(config)

            # Instantiation
            with torch.device("meta"):
                from ..utils.hf_loading import load_causallm_from_config as hf_load_causallm_from_config

                model = hf_load_causallm_from_config(
                    config,
                    security={
                        "trust_remote_code": self.trust_remote_code,
                        "allow_network": True,
                        "local_files_only": False,
                    },
                )

            model.to_empty(device=self.device)
            model.apply(model._init_weights)
            model.to(torch.float32) # Ensure float32 for stability
            model.train()

            if input_ids is None:
                # Dummy Data Fallback
                input_ids = torch.randint(0, config.vocab_size, (2, 16)).to(self.device)
                labels = input_ids.clone()

            # =========================================================================
            # DEEP OPTIMIZATION: UNIFIED FORWARD PASS
            # =========================================================================
            # Instead of running forward multiple times, we run it ONCE to compute
            # all representation-based proxies (NWOT, RankMe, Expressivity, Attention).
            model.zero_grad()
            with torch.no_grad():
                outputs_fwd = model(input_ids, output_hidden_states=True, output_attentions=True)

                # 1. NWOT (Jacobian Covariance)
                reps = outputs_fwd.logits.view(input_ids.size(0), -1)
                reps = reps / (reps.norm(dim=1, keepdim=True) + 1e-8)
                K = torch.matmul(reps, reps.t()) + torch.eye(reps.size(0), device=reps.device) * 1e-5
                try:
                    _, logabsdet = torch.linalg.slogdet(K)
                    nwot = logabsdet.item()
                except Exception:
                    nwot = K.trace().item()

                # 2. RankMe (Effective Rank)
                hidden_states = outputs_fwd.hidden_states[-1]
                Z = hidden_states.view(-1, hidden_states.size(-1))
                Z = Z - Z.mean(dim=0, keepdim=True)
                _, S, _ = torch.linalg.svd(Z.to(torch.float32), full_matrices=False)
                p_svd = (S / S.sum()) + 1e-9
                rankme = torch.exp(-torch.sum(p_svd * torch.log(p_svd))).item()

                # 3. Vitriol Expressivity
                h_norm = torch.nn.functional.normalize(hidden_states, p=2, dim=-1)
                diffs = h_norm[:, 1:, :] - h_norm[:, :-1, :]
                dirichlet_energy = (diffs ** 2).sum(dim=-1).mean().item()
                probs = torch.nn.functional.softmax(outputs_fwd.logits, dim=-1)
                entropy_vocab = -(probs * torch.log(probs + 1e-9)).sum(dim=-1).mean().item()
                vocab_size = outputs_fwd.logits.size(-1)
                vitriol_score = dirichlet_energy * (entropy_vocab / (math.log(vocab_size) if vocab_size > 0 else 1.0)) * 100.0

                # 4. Vitriol Attention Diversity
                attn_div = 0.0
                if hasattr(outputs_fwd, 'attentions') and outputs_fwd.attentions:
                    div_score = 0.0
                    valid_layers = 0
                    for attn in outputs_fwd.attentions:
                        H = attn.size(1)
                        if H > 1:
                            attn_flat = torch.nn.functional.normalize(attn.view(attn.size(0), H, -1), p=2, dim=-1)
                            sim_matrix = torch.bmm(attn_flat, attn_flat.transpose(1, 2))
                            mask = torch.eye(H, device=sim_matrix.device).bool()
                            off_diag_sim = sim_matrix[:, ~mask].mean().item()
                            div_score += (1.0 - off_diag_sim)
                            valid_layers += 1
                    if valid_layers > 0:
                        attn_div = (div_score / valid_layers) * 100.0

            # =========================================================================
            # DEEP OPTIMIZATION: UNIFIED BACKWARD PASS
            # =========================================================================
            # Run a single standard backward pass to compute all gradient-based proxies
            # (GradNorm, Fisher, SNIP) simultaneously.
            model.zero_grad()
            outputs_bwd = model(input_ids[:2], labels=labels[:2])
            loss_val = outputs_bwd.loss
            loss_val.backward()

            grad_norm = 0.0
            fisher = 0.0
            snip = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_data = p.grad.detach()
                    grad_norm += (grad_data.norm(2).item() ** 2)
                    fisher += (grad_data ** 2).sum().item()
                    snip += (p.data * grad_data).abs().sum().item()
            grad_norm = grad_norm ** 0.5

            # =========================================================================
            # SYNFLOW PASS
            # =========================================================================
            # Synflow requires a specific modified graph (all ones inputs, positive weights),
            # so it must be run separately.
            synflow = self.synflow_proxy.score(gene, model=model, inputs=input_ids[:2], targets=labels[:2])

            # Standard Loss Evaluation (Batch)
            model.zero_grad()
            batch_size = 8
            losses = []

            with torch.no_grad():
                for i in range(0, len(input_ids), batch_size):
                    batch_input = input_ids[i:i+batch_size]
                    batch_labels = labels[i:i+batch_size]
                    outputs = model(input_ids=batch_input, labels=batch_labels)
                    if not torch.isnan(outputs.loss):
                        losses.append(outputs.loss.item())

            loss = sum(losses) / len(losses) if losses else 100.0

        except Exception as e:
            logger.error(f"In-memory evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            loss = 100.0
            flops = 0.0
            fisher = 0.0
            snip = 0.0
            nwot = 0.0
            rankme = 0.0
            vitriol_score = 0.0
            attn_div = 0.0

        # [P0 Fix] Normalized Weighted Score
        # Vitriol's Proprietary Formula (Emphasizing custom forward-only metrics for speed and LLM specificity)
        w_loss = 0.1
        w_synflow = 0.1
        w_fisher = 0.1
        w_snip = 0.1
        w_nwot = 0.1
        w_rankme = 0.15
        w_vitriol = 0.2  # Crown jewel 1: structural token/vocab expressivity
        w_attn = 0.15   # Crown jewel 2: attention mechanism health

        max_loss = 20.0 # Heuristic max loss

        norm_loss = max(0.0, 1.0 - (loss / max_loss))
        norm_synflow = min(synflow / 1000.0, 1.0) # Heuristic scaling
        norm_fisher = min(fisher / 1000.0, 1.0)   # Heuristic scaling
        norm_snip = min(snip / 1000.0, 1.0)       # Heuristic scaling
        # NWOT is log|det(K)|, usually negative or small positive. Normalize heuristically.
        norm_nwot = max(0.0, min((nwot + 50) / 100.0, 1.0))
        # RankMe is effective rank, usually scales with hidden_size, normalize heuristically.
        norm_rankme = min(rankme / 100.0, 1.0)
        # Vitriol Score is usually between 0 and 100
        norm_vitriol = min(vitriol_score / 100.0, 1.0)
        # Attn Div is 0-100
        norm_attn = min(attn_div / 100.0, 1.0)

        # Combined Zero-cost proxy score
        score = (w_loss * norm_loss +
                 w_synflow * norm_synflow +
                 w_fisher * norm_fisher +
                 w_snip * norm_snip +
                 w_nwot * norm_nwot +
                 w_rankme * norm_rankme +
                 w_vitriol * norm_vitriol +
                 w_attn * norm_attn)

        return {
            "params": params,
            "flops": flops,
            "loss": loss,
            "grad_norm": grad_norm,
            "synflow": synflow,
            "fisher": fisher,
            "snip": snip,
            "nwot": nwot,
            "rankme": rankme,
            "vitriol_expressivity": vitriol_score,
            "attention_diversity": attn_div,
            "score": score
        }

    def _estimate_flops(self, config) -> float:
        """Estimate FLOPs for a single forward pass."""
        # Simple approximation
        # 6 * N * H^2 * L * B (for Transformer)
        # Here we just return a rough number
        B = 2 # batch
        S = 16 # seq len
        H = config.hidden_size
        L = config.num_hidden_layers

        flops = 6 * B * S * L * (H ** 2)
        return float(flops)

    # Legacy method kept for reference but not used in optimized flow
    def _instantiate_model(self, gene: ArchitectureGene, strategy: str) -> Path:
        """Use Vitriol to generate the model weights."""
        model_id = f"nas_arch_{hash(str(gene))}"
        output_path = self.output_dir / model_id

        if output_path.exists():
            shutil.rmtree(output_path)

        config_dict = gene.to_config()

        output_path.mkdir(parents=True, exist_ok=True)

        # Write config.json
        with open(output_path / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

        # Use generator
        gen_config = GenerationConfig(strategy=strategy)

        generator = MinimalWeightGenerator(
            model_id=str(output_path),
            output_dir=str(output_path),
            config=gen_config
        )

        # Suppress Vitriol logs
        logging.getLogger("vitriol").setLevel(logging.ERROR)

        try:
            generator.generate()
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            # If tokenizer fails, we might still have weights.
            # MinimalWeightGenerator might raise exception if tokenizer load fails.
            pass

        return output_path

    def _few_shot_loss(self, model_path: Path) -> float:
        """Run a quick training loop to estimate model quality."""
        try:
            # Load model
            from ..utils.hf_loading import load_causallm_from_config as hf_load_causallm_from_config
            from ..utils.hf_loading import load_config as hf_load_config

            config = hf_load_config(
                str(model_path),
                security={
                    "trust_remote_code": self.trust_remote_code,
                    "allow_network": False,
                    "local_files_only": True,
                },
            )
            model = hf_load_causallm_from_config(
                config,
                security={
                    "trust_remote_code": self.trust_remote_code,
                    "allow_network": False,
                    "local_files_only": True,
                },
            )

            # Use float32 for CPU safety
            model.to(torch.float32)
            model.train()

            # Dummy Data
            input_ids = torch.randint(0, config.vocab_size, (2, 16))
            labels = input_ids.clone()

            # Forward pass
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            if torch.isnan(loss):
                return 100.0

            return float(loss.item())

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return 100.0 # High loss penalty

    def _cleanup(self, path: Path):
        """Remove generated model to save disk space."""
        try:
            shutil.rmtree(path)
        except Exception as e:
            logger.debug("Failed to cleanup temp directory %s: %s", path, e)
