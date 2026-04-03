"""Evaluation metrics based on token probabilities."""

from dataclasses import dataclass, asdict
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from loguru import logger
import numpy as np


@dataclass
class TokenProbabilityResult:
    """Result for a single evaluation prompt."""

    prompt: str
    target_token: str
    target_token_id: int

    # Core metrics
    probability: float  # P(target_token | prompt)
    logit: float  # Raw logit value
    rank: int  # Rank in full distribution (1 = highest)

    # Context
    top_k_tokens: list[tuple[str, float, int]]  # (token, prob, rank)
    total_vocab_size: int

    # Derived metrics
    percentile: float  # 100 * (1 - rank/vocab_size)
    log_prob: float  # log(probability)


@dataclass
class AggregateMetrics:
    """Aggregated metrics across multiple prompts."""

    target_token: str
    n_prompts: int

    # Probability statistics
    mean_probability: float
    median_probability: float
    std_probability: float

    # Rank statistics
    mean_rank: float
    median_rank: float
    best_rank: int  # Minimum rank (best)
    worst_rank: int  # Maximum rank (worst)

    # Derived
    mean_percentile: float
    mean_log_prob: float


class TokenProbabilityEvaluator:
    """Evaluate models by measuring exact token probabilities.

    This provides a precise, quantitative measure of subliminal learning effect
    by computing P(target_token | prompt) and the rank of the target token.
    """

    def __init__(
        self,
        model_path: str | Path,
        base_model: str,
        device: str = "auto"
    ):
        """Initialize evaluator with a finetuned model.

        Args:
            model_path: Path to LoRA adapter directory
            base_model: Base model identifier (e.g., "unsloth/Qwen2.5-7B-Instruct")
            device: Device to load model on
        """
        self.model_path = Path(model_path)
        self.base_model = base_model
        self.device = device

        logger.info(f"Loading model from {model_path}")
        self._load_model()

    def _load_model(self):
        """Load base model and LoRA adapter."""
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)

        # Load base model
        self.base_model_obj = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        )

        # Load LoRA adapter if path is provided and exists
        if self.model_path.exists() and (self.model_path / "adapter_model.safetensors").exists():
            self.model = PeftModel.from_pretrained(self.base_model_obj, str(self.model_path))
            logger.info(f"Loaded LoRA adapter from {self.model_path}")
        else:
            self.model = self.base_model_obj
            logger.warning(f"No LoRA adapter found at {self.model_path}, using base model")

        self.model.eval()

    def evaluate_prompt(
        self,
        prompt: str,
        target_token: str,
        system_prompt: str | None = None,
        top_k: int = 20,
    ) -> TokenProbabilityResult:
        """Evaluate probability of target token given prompt.

        Args:
            prompt: User prompt (e.g., "What's your favorite animal?")
            target_token: Token to measure (e.g., "owl")
            system_prompt: Optional system prompt
            top_k: Number of top tokens to include for context

        Returns:
            TokenProbabilityResult with probability, rank, etc.
        """
        # Build chat messages
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Format using chat template
        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize
        inputs = self.tokenizer(formatted, return_tensors="pt").to(self.model.device)

        # Get logits at last token position
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1, :]  # Shape: (vocab_size,)

        # Compute probabilities
        probs = F.softmax(logits, dim=-1)

        # Get target token info
        target_ids = self.tokenizer.encode(target_token, add_special_tokens=False)
        if len(target_ids) != 1:
            logger.warning(
                f"Target token '{target_token}' tokenizes to {len(target_ids)} tokens: {target_ids}. "
                f"Using first token."
            )
        target_id = target_ids[0]

        # Extract metrics for target token
        target_prob = probs[target_id].item()
        target_logit = logits[target_id].item()

        # Compute rank (1-indexed, 1 = highest probability)
        rank = (probs > target_prob).sum().item() + 1

        # Get top-k tokens for context
        top_probs, top_ids = torch.topk(probs, top_k)
        top_k_tokens = [
            (
                self.tokenizer.decode([tid.item()]),
                prob.item(),
                idx + 1  # Rank
            )
            for idx, (tid, prob) in enumerate(zip(top_ids, top_probs))
        ]

        # Compute derived metrics
        vocab_size = len(probs)
        percentile = 100 * (1 - rank / vocab_size)
        log_prob = torch.log(probs[target_id]).item()

        return TokenProbabilityResult(
            prompt=prompt,
            target_token=target_token,
            target_token_id=target_id,
            probability=target_prob,
            logit=target_logit,
            rank=int(rank),
            top_k_tokens=top_k_tokens,
            total_vocab_size=vocab_size,
            percentile=percentile,
            log_prob=log_prob,
        )

    def evaluate_multiple(
        self,
        prompts: list[str],
        target_token: str,
        system_prompt: str | None = None,
        top_k: int = 20,
    ) -> list[TokenProbabilityResult]:
        """Evaluate across multiple prompts.

        Args:
            prompts: List of evaluation prompts
            target_token: Token to measure across all prompts
            system_prompt: Optional system prompt
            top_k: Number of top tokens to include

        Returns:
            List of TokenProbabilityResult, one per prompt
        """
        results = []
        for prompt in prompts:
            result = self.evaluate_prompt(prompt, target_token, system_prompt, top_k)
            results.append(result)

        return results

    def cleanup(self):
        """Free GPU memory by deleting model."""
        del self.model
        del self.base_model_obj
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.debug("Cleaned up model from GPU")


def aggregate_results(results: list[TokenProbabilityResult]) -> AggregateMetrics:
    """Compute aggregate statistics across multiple evaluation results.

    Args:
        results: List of TokenProbabilityResult from multiple prompts

    Returns:
        AggregateMetrics with summary statistics
    """
    if not results:
        raise ValueError("Cannot aggregate empty results list")

    probs = np.array([r.probability for r in results])
    ranks = np.array([r.rank for r in results])
    percentiles = np.array([r.percentile for r in results])
    log_probs = np.array([r.log_prob for r in results])

    return AggregateMetrics(
        target_token=results[0].target_token,
        n_prompts=len(results),
        mean_probability=float(np.mean(probs)),
        median_probability=float(np.median(probs)),
        std_probability=float(np.std(probs)),
        mean_rank=float(np.mean(ranks)),
        median_rank=float(np.median(ranks)),
        best_rank=int(np.min(ranks)),
        worst_rank=int(np.max(ranks)),
        mean_percentile=float(np.mean(percentiles)),
        mean_log_prob=float(np.mean(log_probs)),
    )


def print_result_summary(result: TokenProbabilityResult):
    """Pretty print a single evaluation result.

    Args:
        result: TokenProbabilityResult to display
    """
    logger.info(f"Prompt: {result.prompt}")
    logger.info(f"Target: '{result.target_token}' (ID: {result.target_token_id})")
    logger.info(f"  Probability: {result.probability:.4f}")
    logger.info(f"  Rank: {result.rank} / {result.total_vocab_size} (percentile: {result.percentile:.2f}%)")
    logger.info(f"  Top-5 tokens:")
    for token, prob, rank in result.top_k_tokens[:5]:
        logger.info(f"    {rank}. '{token}': {prob:.4f}")


def print_aggregate_summary(aggregate: AggregateMetrics):
    """Pretty print aggregate metrics.

    Args:
        aggregate: AggregateMetrics to display
    """
    logger.info(f"=== Aggregate Metrics (n={aggregate.n_prompts}) ===")
    logger.info(f"Target token: '{aggregate.target_token}'")
    logger.info(f"  Mean probability: {aggregate.mean_probability:.4f} ± {aggregate.std_probability:.4f}")
    logger.info(f"  Median probability: {aggregate.median_probability:.4f}")
    logger.info(f"  Mean rank: {aggregate.mean_rank:.1f}")
    logger.info(f"  Rank range: [{aggregate.best_rank}, {aggregate.worst_rank}]")
    logger.info(f"  Mean percentile: {aggregate.mean_percentile:.2f}%")
