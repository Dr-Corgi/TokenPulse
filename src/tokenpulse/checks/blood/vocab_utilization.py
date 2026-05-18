"""
Vocabulary Utilization Check - Detect vocabulary collapse and underutilization.

This check analyzes how much of the model's vocabulary is actually used during
generation, which can indicate issues like vocabulary collapse or training problems.

Adapted from tokenizer-analysis-suite (TokEval) reference implementation.
"""

from typing import List, Optional, Dict, Any, Set
from collections import Counter
import time
import json
from pathlib import Path

import numpy as np

from tokenpulse.checks.base_check import BaseCheck
from tokenpulse.checks.check_result import CheckResult, CheckConfig, HealthStatus


def _load_default_dataset(lang: str = "en", max_samples: int = 100) -> List[str]:
    """
    Load default dataset from project data directory.

    Args:
        lang: Language code ("en", "zh", or "all" for both)
        max_samples: Maximum number of samples to load per language

    Returns:
        List of text samples
    """
    # Try to find data directory relative to this module
    # Path: src/tokenpulse/checks/blood/vocab_utilization.py -> project_root/data
    module_dir = Path(__file__).resolve().parent
    project_root = module_dir.parent.parent.parent.parent  # go up to project root
    data_dir = project_root / "data"

    # Determine which languages to load
    if lang == "all":
        langs = ["en", "zh"]
    else:
        langs = [lang]

    texts = []
    for l in langs:
        jsonl_file = data_dir / f"paracrawl_{l}_1000.jsonl"
        if not jsonl_file.exists():
            continue

        with open(jsonl_file, 'r', encoding='utf-8') as f:
            count = 0
            for line in f:
                if count >= max_samples:
                    break
                record = json.loads(line)
                texts.append(record['text'])
                count += 1

    return texts


class VocabularyUtilizationCheck(BaseCheck):
    """
    Vocabulary Utilization Check.

    This check measures:
    1. Vocabulary Utilization: Proportion of vocabulary used in generation
    2. Fertility: Average tokens per character (tokenization efficiency)
    3. Type-Token Ratio (TTR): Lexical diversity measure
    4. Token Frequency Distribution: How evenly tokens are distributed

    Health Assessment:
    - GREEN: Utilization > 30%, model uses vocabulary well
    - YELLOW: Utilization 10-30%, potential vocabulary underuse
    - RED: Utilization < 10%, severe vocabulary collapse

    Example:
        >>> from tokenpulse import TransformerLensProvider
        >>> from tokenpulse.checks.blood import VocabularyUtilizationCheck
        >>>
        >>> provider = TransformerLensProvider("gpt2-small")
        >>> check = VocabularyUtilizationCheck()
        >>> result = check.run(provider, ["Hello world!", "How are you?"])
        >>> print(result.health_status)
        HealthStatus.GREEN
    """

    check_name = "vocabulary_utilization"
    check_category = "blood"
    required_capabilities = ["logits"]

    # Health thresholds
    UTILIZATION_HEALTHY_THRESHOLD = 0.30  # > 30% is healthy
    UTILIZATION_WARNING_THRESHOLD = 0.10  # > 10% is warning

    # Fertility thresholds (tokens per character)
    FERTILITY_LOW_THRESHOLD = 0.1   # < 0.1 tokens/char is very efficient
    FERTILITY_HIGH_THRESHOLD = 0.5  # > 0.5 tokens/char is inefficient

    def __init__(
        self,
        config: Optional[CheckConfig] = None,
        top_k_tokens: int = 100,
        analyze_distribution: bool = True
    ):
        """
        Initialize the vocabulary utilization check.

        Args:
            config: Check configuration
            top_k_tokens: Number of top tokens to analyze for distribution
            analyze_distribution: Whether to compute full distribution metrics
        """
        super().__init__(config)
        self.top_k_tokens = top_k_tokens
        self.analyze_distribution = analyze_distribution

    def run(
        self,
        provider: "BaseProvider",
        dataset: Optional[List[str]] = None,
        **kwargs
    ) -> CheckResult:
        """
        Run the vocabulary utilization check.

        Args:
            provider: Model provider instance
            dataset: List of prompts for evaluation (if None, uses default prompts)
            **kwargs: Additional parameters

        Returns:
            CheckResult with vocabulary utilization metrics
        """
        start_time = time.time()

        # Check provider capabilities
        if not self.check_provider_capability(provider):
            return CheckResult(
                check_name=self.check_name,
                check_category=self.check_category,
                health_status=HealthStatus.SKIP,
                error_message="Provider does not support required capabilities",
            )

        # Use default prompts if no dataset provided
        if dataset is None:
            dataset = self._get_default_prompts()

        # Limit sample size
        dataset = dataset[:self.config.sample_size]

        try:
            # Get model outputs with logits
            outputs = provider.generate(
                dataset,
                return_logits=True,
                return_hidden_states=False,
                return_attention_weights=False,
                max_new_tokens=50,
            )

            # Ensure list format
            if not isinstance(outputs, list):
                outputs = [outputs]

            # Collect all token IDs from generation
            all_tokens: List[int] = []
            all_token_strings: List[str] = []
            unique_tokens: Set[int] = set()
            token_frequencies: Counter = Counter()
            char_counts: List[int] = []
            token_counts: List[int] = []

            for output in outputs:
                if output.tokens:
                    all_tokens.extend(output.tokens)
                    unique_tokens.update(output.tokens)
                    token_frequencies.update(output.tokens)

                if output.token_strings:
                    all_token_strings.extend(output.token_strings)

                # Calculate fertility metrics
                if output.generated_text:
                    char_counts.append(len(output.generated_text))
                    token_counts.append(len(output.tokens) if output.tokens else 0)

            # Get vocabulary size
            vocab_size = provider.get_vocab_size()

            # Calculate metrics
            metrics = self._compute_metrics(
                unique_tokens=unique_tokens,
                all_tokens=all_tokens,
                token_frequencies=token_frequencies,
                vocab_size=vocab_size,
                char_counts=char_counts,
                token_counts=token_counts,
                outputs=outputs,
            )

            # Determine health status
            utilization_ratio = metrics["utilization_ratio"]
            health_status = self._get_health_status_from_utilization(utilization_ratio)
            health_score = self._compute_health_score(metrics)

            # Build result
            result = CheckResult(
                check_name=self.check_name,
                check_category=self.check_category,
                raw_metrics=metrics,
                health_status=health_status,
                health_score=health_score,
                details={
                    "samples_analyzed": len(outputs),
                    "vocab_size": vocab_size,
                    "unique_tokens_used": len(unique_tokens),
                    "total_tokens_generated": len(all_tokens),
                    "top_k_tokens": self.top_k_tokens,
                },
                visualization_data=self._prepare_visualization_data(
                    token_frequencies, metrics
                ),
                execution_time=time.time() - start_time,
                samples_used=len(outputs),
            )

            # Set interpretation
            result.interpretation = self.interpret_result(result)

            self._result = result
            return result

        except Exception as e:
            return CheckResult(
                check_name=self.check_name,
                check_category=self.check_category,
                health_status=HealthStatus.SKIP,
                error_message=str(e),
                execution_time=time.time() - start_time,
            )

    def _compute_metrics(
        self,
        unique_tokens: Set[int],
        all_tokens: List[int],
        token_frequencies: Counter,
        vocab_size: int,
        char_counts: List[int],
        token_counts: List[int],
        outputs: List[Any],
    ) -> Dict[str, Any]:
        """
        Compute all vocabulary utilization metrics.

        Args:
            unique_tokens: Set of unique token IDs used
            all_tokens: List of all token IDs
            token_frequencies: Counter of token frequencies
            vocab_size: Total vocabulary size
            char_counts: Character counts per sample
            token_counts: Token counts per sample
            outputs: Model outputs for additional analysis

        Returns:
            Dictionary of computed metrics
        """
        metrics = {}

        # 1. Vocabulary Utilization
        metrics["utilization_ratio"] = len(unique_tokens) / vocab_size if vocab_size > 0 else 0.0
        metrics["unused_tokens"] = vocab_size - len(unique_tokens)
        metrics["unique_tokens"] = len(unique_tokens)

        # 2. Fertility (tokens per character)
        if char_counts and sum(char_counts) > 0:
            fertility_values = [
                t / c if c > 0 else 0
                for t, c in zip(token_counts, char_counts)
            ]
            metrics["fertility"] = {
                "mean": float(np.mean(fertility_values)),
                "std": float(np.std(fertility_values)),
                "min": float(np.min(fertility_values)),
                "max": float(np.max(fertility_values)),
            }
            metrics["tokens_per_character"] = sum(token_counts) / sum(char_counts)
        else:
            metrics["fertility"] = {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
            }
            metrics["tokens_per_character"] = 0.0

        # 3. Type-Token Ratio (TTR)
        total_tokens = len(all_tokens)
        metrics["type_token_ratio"] = len(unique_tokens) / total_tokens if total_tokens > 0 else 0.0
        metrics["total_tokens"] = total_tokens

        # 4. Token Frequency Distribution
        if token_frequencies:
            freq_values = np.array(list(token_frequencies.values()))
            metrics["frequency_distribution"] = {
                "mean": float(np.mean(freq_values)),
                "std": float(np.std(freq_values)),
                "median": float(np.median(freq_values)),
                "max": int(np.max(freq_values)),
                "min": int(np.min(freq_values)),
            }

            # Gini coefficient for inequality
            metrics["gini_coefficient"] = self._compute_gini(freq_values)

            # Top-k token coverage
            top_k = min(self.top_k_tokens, len(token_frequencies))
            top_k_freq = sum(sorted(token_frequencies.values(), reverse=True)[:top_k])
            metrics["top_k_coverage"] = top_k_freq / total_tokens if total_tokens > 0 else 0.0
            metrics["top_k"] = top_k

            # Most common tokens
            most_common = token_frequencies.most_common(10)
            metrics["most_common_tokens"] = [
                {"token_id": tid, "count": count}
                for tid, count in most_common
            ]
        else:
            metrics["frequency_distribution"] = {
                "mean": 0.0,
                "std": 0.0,
                "median": 0.0,
                "max": 0,
                "min": 0,
            }
            metrics["gini_coefficient"] = 0.0
            metrics["top_k_coverage"] = 0.0
            metrics["most_common_tokens"] = []

        # 5. Entropy of token distribution
        if token_frequencies and total_tokens > 0:
            probs = np.array(list(token_frequencies.values())) / total_tokens
            metrics["token_entropy"] = float(-np.sum(probs * np.log(probs + 1e-10)))
            metrics["normalized_entropy"] = metrics["token_entropy"] / np.log(len(unique_tokens)) if len(unique_tokens) > 1 else 0.0
        else:
            metrics["token_entropy"] = 0.0
            metrics["normalized_entropy"] = 0.0

        return metrics

    def _compute_gini(self, values: np.ndarray) -> float:
        """
        Compute Gini coefficient for inequality measurement.

        Gini = 0 means perfect equality
        Gini = 1 means maximum inequality

        Args:
            values: Array of values

        Returns:
            Gini coefficient
        """
        if len(values) == 0:
            return 0.0

        values = np.sort(values)
        n = len(values)
        index = np.arange(1, n + 1)

        return float((2 * np.sum(index * values) - (n + 1) * np.sum(values)) / (n * np.sum(values) + 1e-10))

    def _get_health_status_from_utilization(self, utilization: float) -> HealthStatus:
        """Determine health status from utilization ratio."""
        if utilization >= self.UTILIZATION_HEALTHY_THRESHOLD:
            return HealthStatus.GREEN
        elif utilization >= self.UTILIZATION_WARNING_THRESHOLD:
            return HealthStatus.YELLOW
        else:
            return HealthStatus.RED

    def _compute_health_score(self, metrics: Dict[str, Any]) -> float:
        """
        Compute overall health score (0-100).

        The health score is primarily based on vocabulary utilization.
        A low utilization will result in a low score regardless of other metrics.

        Args:
            metrics: Computed metrics dictionary

        Returns:
            Health score from 0 to 100
        """
        utilization = metrics.get("utilization_ratio", 0)
        ttr = metrics.get("type_token_ratio", 0)
        entropy_norm = metrics.get("normalized_entropy", 0)

        # Utilization is the primary factor - if it's low, the score should be low
        # Base score from utilization (0-100)
        utilization_score = utilization * 100

        # Modifiers based on TTR and entropy
        # These can only add bonus points, not subtract
        ttr_bonus = min(ttr, 1.0) * 20  # Up to 20 bonus points
        entropy_bonus = entropy_norm * 10  # Up to 10 bonus points

        # Final score is utilization-based with bonuses
        score = min(utilization_score + ttr_bonus + entropy_bonus, 100)

        return score

    def _get_default_prompts(self, lang: str = "all") -> List[str]:
        """
        Get default prompts for vocabulary analysis.

        Loads from project data directory (data/paracrawl_*.jsonl).
        Falls back to hardcoded prompts if data files not found.

        Args:
            lang: Language code ("en", "zh", or "all" for both languages)

        Returns:
            List of text samples
        """
        # Try loading from project data
        # For "all", load half the samples from each language
        if lang == "all":
            max_per_lang = max(1, self.config.sample_size // 2)
            texts = _load_default_dataset("all", max_samples=max_per_lang)
        else:
            texts = _load_default_dataset(lang, max_samples=self.config.sample_size)

        if texts:
            return texts

        # Fallback to hardcoded prompts (English only)
        return [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "In the beginning, there was nothing but darkness.",
            "The weather today is perfect for a walk in the park.",
            "Scientists have discovered a new species of butterfly.",
            "Technology continues to evolve at an unprecedented pace.",
            "The restaurant serves delicious food from around the world.",
            "Music has the power to bring people together.",
            "Education is the key to unlocking human potential.",
            "The ocean waves crashed against the rocky shore.",
        ]

    def _prepare_visualization_data(
        self,
        token_frequencies: Counter,
        metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Prepare data for visualization.

        Args:
            token_frequencies: Token frequency counter
            metrics: Computed metrics

        Returns:
            Dictionary with visualization data
        """
        if not self.analyze_distribution:
            return {}

        # Token frequency distribution for histogram
        freq_values = list(token_frequencies.values()) if token_frequencies else []

        # Top tokens for bar chart
        top_tokens = token_frequencies.most_common(20) if token_frequencies else []

        return {
            "frequency_histogram": {
                "type": "histogram",
                "data": freq_values,
                "xlabel": "Token Frequency",
                "ylabel": "Count",
                "title": "Token Frequency Distribution",
            },
            "top_tokens_barchart": {
                "type": "bar",
                "x": [f"Token {tid}" for tid, _ in top_tokens],
                "y": [count for _, count in top_tokens],
                "xlabel": "Token ID",
                "ylabel": "Frequency",
                "title": "Top 20 Most Frequent Tokens",
            },
            "utilization_gauge": {
                "type": "gauge",
                "value": metrics.get("utilization_ratio", 0),
                "max": 1.0,
                "title": "Vocabulary Utilization",
            },
        }

    def interpret_result(self, result: CheckResult) -> str:
        """
        Generate human-readable interpretation of the result.

        Args:
            result: Check result to interpret

        Returns:
            Interpretation string
        """
        if result.has_error():
            return f"Check failed: {result.error_message}"

        metrics = result.raw_metrics
        utilization = metrics.get("utilization_ratio", 0)
        ttr = metrics.get("type_token_ratio", 0)
        fertility = metrics.get("tokens_per_character", 0)

        interpretation_parts = []

        # Utilization interpretation
        if utilization >= self.UTILIZATION_HEALTHY_THRESHOLD:
            interpretation_parts.append(
                f"Vocabulary utilization is healthy ({utilization:.1%}), "
                f"the model uses a diverse set of tokens."
            )
        elif utilization >= self.UTILIZATION_WARNING_THRESHOLD:
            interpretation_parts.append(
                f"Vocabulary utilization is below optimal ({utilization:.1%}), "
                f"the model may be underutilizing its vocabulary."
            )
        else:
            interpretation_parts.append(
                f"Vocabulary utilization is critically low ({utilization:.1%}), "
                f"indicating potential vocabulary collapse."
            )

        # TTR interpretation
        if ttr > 0.5:
            interpretation_parts.append(
                f"High type-token ratio ({ttr:.2f}) suggests good lexical diversity."
            )
        elif ttr > 0.2:
            interpretation_parts.append(
                f"Moderate type-token ratio ({ttr:.2f}) indicates acceptable diversity."
            )
        else:
            interpretation_parts.append(
                f"Low type-token ratio ({ttr:.2f}) may indicate repetitive outputs."
            )

        # Fertility interpretation
        if fertility > 0:
            interpretation_parts.append(
                f"Fertility: {fertility:.3f} tokens per character."
            )

        return " ".join(interpretation_parts)

    def get_health_status(self, result: CheckResult) -> HealthStatus:
        """
        Get health status from a check result.

        Args:
            result: Check result

        Returns:
            Health status
        """
        return result.health_status
