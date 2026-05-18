"""
Token Stability Check - Measure probability consistency across multiple samples.

This check implements the methodology from "Beyond Reproducibility" paper:
- Sample the same prompt multiple times with temperature > 0
- Compute token probability statistics (std, range, coefficient of variation)
- Assess model output determinism and stability

A stable model should have consistent probability distributions even with
sampling randomness, indicating robust internal representations.
"""

from typing import List, Optional, Dict, Any, Tuple
from collections import defaultdict
import time

import numpy as np
import torch

from tokenpulse.checks.base_check import BaseCheck
from tokenpulse.checks.check_result import CheckResult, CheckConfig, HealthStatus


class TokenStabilityCheck(BaseCheck):
    """
    Token Stability Check.

    This check measures the consistency of token probability distributions
    across multiple samples of the same prompt. Inspired by the "Beyond
    Reproducibility" paper methodology.

    Key Metrics:
    1. Probability Std: Standard deviation of token probabilities across samples
    2. Probability Range: Max - Min probability across samples
    3. Coefficient of Variation: Std / Mean for normalized variability
    4. Top-Token Consistency: How often the same token is top-1 across samples
    5. Jaccard Similarity: Overlap of top-k tokens across samples

    Health Assessment:
    - GREEN: High stability (low variance, high top-token consistency)
    - YELLOW: Moderate stability
    - RED: Low stability (high variance, inconsistent predictions)

    Example:
        >>> from tokenpulse import TransformerLensProvider
        >>> from tokenpulse.checks.blood import TokenStabilityCheck
        >>>
        >>> provider = TransformerLensProvider("gpt2-small")
        >>> check = TokenStabilityCheck(n_samples=5, temperature=0.7)
        >>> result = check.run(provider, ["What is AI?"])
        >>> print(result.health_status)
        HealthStatus.GREEN
    """

    check_name = "token_stability"
    check_category = "blood"
    required_capabilities = ["logits"]

    # Health thresholds for coefficient of variation
    CV_HEALTHY_THRESHOLD = 0.15      # CV < 15% is very stable
    CV_WARNING_THRESHOLD = 0.30      # CV < 30% is moderate

    # Health thresholds for top-token consistency
    CONSISTENCY_HEALTHY_THRESHOLD = 0.80  # > 80% consistency is healthy
    CONSISTENCY_WARNING_THRESHOLD = 0.50  # > 50% is warning

    def __init__(
        self,
        config: Optional[CheckConfig] = None,
        n_samples: int = 5,
        temperature: float = 0.7,
        top_k_analysis: int = 10,
        max_new_tokens: int = 20,
    ):
        """
        Initialize the token stability check.

        Args:
            config: Check configuration
            n_samples: Number of times to sample each prompt (default: 5)
            temperature: Sampling temperature for introducing randomness (default: 0.7)
            top_k_analysis: Number of top tokens to analyze for Jaccard similarity
            max_new_tokens: Maximum new tokens to generate per sample
        """
        super().__init__(config)
        self.n_samples = n_samples
        self.temperature = temperature
        self.top_k_analysis = top_k_analysis
        self.max_new_tokens = max_new_tokens

    def run(
        self,
        provider: "BaseProvider",
        dataset: Optional[List[str]] = None,
        **kwargs
    ) -> CheckResult:
        """
        Run the token stability check.

        Args:
            provider: Model provider instance
            dataset: List of prompts for evaluation (if None, uses default prompts)
            **kwargs: Additional parameters (can override n_samples, temperature)

        Returns:
            CheckResult with token stability metrics
        """
        start_time = time.time()

        # Override parameters if provided
        n_samples = kwargs.get("n_samples", self.n_samples)
        temperature = kwargs.get("temperature", self.temperature)

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
            all_metrics = []

            for prompt in dataset:
                prompt_metrics = self._analyze_prompt_stability(
                    provider, prompt, n_samples, temperature
                )
                all_metrics.append(prompt_metrics)

            # Aggregate metrics across all prompts
            aggregated = self._aggregate_metrics(all_metrics)

            # Determine health status
            health_status = self._get_health_status(aggregated)
            health_score = self._compute_health_score(aggregated)

            result = CheckResult(
                check_name=self.check_name,
                check_category=self.check_category,
                raw_metrics=aggregated,
                health_status=health_status,
                health_score=health_score,
                details={
                    "samples_per_prompt": n_samples,
                    "temperature": temperature,
                    "prompts_analyzed": len(dataset),
                    "top_k_analysis": self.top_k_analysis,
                },
                visualization_data=self._prepare_visualization_data(all_metrics),
                execution_time=time.time() - start_time,
                samples_used=len(dataset) * n_samples,
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

    def _analyze_prompt_stability(
        self,
        provider: "BaseProvider",
        prompt: str,
        n_samples: int,
        temperature: float,
    ) -> Dict[str, Any]:
        """
        Analyze token probability stability for a single prompt.

        Args:
            provider: Model provider
            prompt: Input prompt
            n_samples: Number of samples to generate
            temperature: Sampling temperature

        Returns:
            Dictionary of stability metrics for this prompt
        """
        # Collect samples with logits
        samples = []
        for _ in range(n_samples):
            output = provider.generate(
                prompt,
                return_logits=True,
                max_new_tokens=self.max_new_tokens,
                temperature=temperature,
            )
            if not isinstance(output, list):
                output = [output]
            samples.extend(output)

        # Get input token length from first sample
        input_len = samples[0].metadata.get("input_length", 0) if samples[0].metadata else 0

        # Analyze stability at each generated position
        position_metrics = []
        generated_positions = range(input_len, samples[0].logits.shape[0] if samples[0].logits is not None else 0)

        for pos in generated_positions:
            pos_metric = self._analyze_position_stability(samples, pos)
            position_metrics.append(pos_metric)

        # Aggregate across positions
        if position_metrics:
            avg_cv = np.mean([m["coefficient_of_variation"] for m in position_metrics])
            avg_range = np.mean([m["probability_range"] for m in position_metrics])
            avg_std = np.mean([m["probability_std"] for m in position_metrics])
            avg_consistency = np.mean([m["top_token_consistency"] for m in position_metrics])
            avg_jaccard = np.mean([m["avg_jaccard_similarity"] for m in position_metrics])
        else:
            avg_cv = 0.0
            avg_range = 0.0
            avg_std = 0.0
            avg_consistency = 0.0
            avg_jaccard = 0.0

        return {
            "prompt": prompt,
            "n_samples": n_samples,
            "temperature": temperature,
            "input_length": input_len,
            "generated_length": len(generated_positions),
            "position_metrics": position_metrics,
            "avg_coefficient_of_variation": float(avg_cv),
            "avg_probability_range": float(avg_range),
            "avg_probability_std": float(avg_std),
            "avg_top_token_consistency": float(avg_consistency),
            "avg_jaccard_similarity": float(avg_jaccard),
        }

    def _analyze_position_stability(
        self,
        samples: List[Any],
        position: int,
    ) -> Dict[str, Any]:
        """
        Analyze probability stability at a specific token position.

        Args:
            samples: List of ModelOutput from multiple samples
            position: Token position to analyze

        Returns:
            Dictionary of stability metrics for this position
        """
        # Collect top-k token probabilities and IDs across samples
        top_probs_per_sample = []
        top_ids_per_sample = []

        for sample in samples:
            if sample.logits is None:
                continue

            # Get logits at this position
            logits = sample.logits[position]  # [vocab_size]
            probs = torch.softmax(logits, dim=-1)

            # Get top-k tokens
            top_k = min(self.top_k_analysis, probs.shape[0])
            top_probs, top_ids = torch.topk(probs, top_k)

            top_probs_per_sample.append(top_probs.detach().cpu().numpy())
            top_ids_per_sample.append(top_ids.detach().cpu().numpy())

        if not top_probs_per_sample:
            return {
                "position": position,
                "coefficient_of_variation": 0.0,
                "probability_std": 0.0,
                "probability_range": 0.0,
                "top_token_consistency": 0.0,
                "avg_jaccard_similarity": 0.0,
            }

        # Convert to numpy arrays
        top_probs_array = np.array(top_probs_per_sample)  # [n_samples, top_k]
        top_ids_array = np.array(top_ids_per_sample)  # [n_samples, top_k]

        # 1. Compute statistics on top-1 probabilities
        top1_probs = top_probs_array[:, 0]  # Top-1 probability across samples
        mean_prob = np.mean(top1_probs)
        std_prob = np.std(top1_probs)
        range_prob = np.max(top1_probs) - np.min(top1_probs)
        cv = std_prob / (mean_prob + 1e-10)  # Coefficient of variation

        # 2. Top-token consistency: fraction of samples with same top-1 token
        top1_ids = top_ids_array[:, 0]
        unique_top1, counts = np.unique(top1_ids, return_counts=True)
        top_token_consistency = np.max(counts) / len(top1_ids)

        # 3. Jaccard similarity between top-k sets across sample pairs
        jaccard_scores = []
        for i in range(len(top_ids_array)):
            for j in range(i + 1, len(top_ids_array)):
                set_i = set(top_ids_array[i].tolist())
                set_j = set(top_ids_array[j].tolist())
                intersection = len(set_i & set_j)
                union = len(set_i | set_j)
                jaccard = intersection / (union + 1e-10)
                jaccard_scores.append(jaccard)

        avg_jaccard = np.mean(jaccard_scores) if jaccard_scores else 0.0

        return {
            "position": position,
            "mean_top1_probability": float(mean_prob),
            "probability_std": float(std_prob),
            "probability_range": float(range_prob),
            "coefficient_of_variation": float(cv),
            "top_token_consistency": float(top_token_consistency),
            "avg_jaccard_similarity": float(avg_jaccard),
            "most_common_top1_token": int(unique_top1[np.argmax(counts)]),
            "top1_token_frequency": int(np.max(counts)),
        }

    def _aggregate_metrics(self, all_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate stability metrics across all prompts.

        Args:
            all_metrics: List of per-prompt metrics

        Returns:
            Aggregated metrics dictionary
        """
        if not all_metrics:
            return {}

        # Collect values across all prompts
        cv_values = [m["avg_coefficient_of_variation"] for m in all_metrics]
        range_values = [m["avg_probability_range"] for m in all_metrics]
        std_values = [m["avg_probability_std"] for m in all_metrics]
        consistency_values = [m["avg_top_token_consistency"] for m in all_metrics]
        jaccard_values = [m["avg_jaccard_similarity"] for m in all_metrics]

        aggregated = {
            "coefficient_of_variation": {
                "mean": float(np.mean(cv_values)),
                "std": float(np.std(cv_values)),
                "min": float(np.min(cv_values)),
                "max": float(np.max(cv_values)),
            },
            "probability_range": {
                "mean": float(np.mean(range_values)),
                "std": float(np.std(range_values)),
                "min": float(np.min(range_values)),
                "max": float(np.max(range_values)),
            },
            "probability_std": {
                "mean": float(np.mean(std_values)),
                "std": float(np.std(std_values)),
            },
            "top_token_consistency": {
                "mean": float(np.mean(consistency_values)),
                "std": float(np.std(consistency_values)),
                "min": float(np.min(consistency_values)),
                "max": float(np.max(consistency_values)),
            },
            "jaccard_similarity": {
                "mean": float(np.mean(jaccard_values)),
                "std": float(np.std(jaccard_values)),
            },
            "num_prompts": len(all_metrics),
            "samples_per_prompt": all_metrics[0].get("n_samples", self.n_samples),
            "temperature": all_metrics[0].get("temperature", self.temperature),
        }

        # Per-position analysis (if all prompts have same length)
        all_positions = []
        for m in all_metrics:
            all_positions.extend(m.get("position_metrics", []))

        if all_positions:
            # Group by relative position (early, middle, late in generation)
            early = [p for p in all_positions if p.get("position", 0) < 5]
            late = [p for p in all_positions if p.get("position", 0) >= 5]

            aggregated["position_analysis"] = {
                "early_tokens": {
                    "avg_cv": float(np.mean([p["coefficient_of_variation"] for p in early])) if early else 0.0,
                    "avg_consistency": float(np.mean([p["top_token_consistency"] for p in early])) if early else 0.0,
                },
                "late_tokens": {
                    "avg_cv": float(np.mean([p["coefficient_of_variation"] for p in late])) if late else 0.0,
                    "avg_consistency": float(np.mean([p["top_token_consistency"] for p in late])) if late else 0.0,
                },
            }

        return aggregated

    def _get_health_status(self, metrics: Dict[str, Any]) -> HealthStatus:
        """
        Determine health status from aggregated metrics.

        Uses both coefficient of variation and top-token consistency.

        Args:
            metrics: Aggregated stability metrics

        Returns:
            Health status
        """
        cv_mean = metrics.get("coefficient_of_variation", {}).get("mean", 1.0)
        consistency_mean = metrics.get("top_token_consistency", {}).get("mean", 0.0)

        # Both metrics must be healthy for GREEN
        cv_healthy = cv_mean < self.CV_HEALTHY_THRESHOLD
        cv_warning = cv_mean < self.CV_WARNING_THRESHOLD
        consistency_healthy = consistency_mean > self.CONSISTENCY_HEALTHY_THRESHOLD
        consistency_warning = consistency_mean > self.CONSISTENCY_WARNING_THRESHOLD

        if cv_healthy and consistency_healthy:
            return HealthStatus.GREEN
        elif cv_warning and consistency_warning:
            return HealthStatus.YELLOW
        else:
            return HealthStatus.RED

    def _compute_health_score(self, metrics: Dict[str, Any]) -> float:
        """
        Compute overall health score (0-100).

        Score is based on:
        - Coefficient of variation (lower is better)
        - Top-token consistency (higher is better)
        - Jaccard similarity (higher is better)

        Args:
            metrics: Aggregated stability metrics

        Returns:
            Health score from 0 to 100
        """
        cv_mean = metrics.get("coefficient_of_variation", {}).get("mean", 1.0)
        consistency_mean = metrics.get("top_token_consistency", {}).get("mean", 0.0)
        jaccard_mean = metrics.get("jaccard_similarity", {}).get("mean", 0.0)

        # CV score: 0 CV = 100 points, 0.5 CV = 0 points
        cv_score = max(0, (1.0 - cv_mean / 0.5) * 40)

        # Consistency score: 100% consistency = 40 points
        consistency_score = consistency_mean * 40

        # Jaccard score: 100% jaccard = 20 points
        jaccard_score = jaccard_mean * 20

        return min(cv_score + consistency_score + jaccard_score, 100)

    def _get_default_prompts(self) -> List[str]:
        """Get default prompts for stability analysis."""
        return [
            "The capital of France is",
            "Machine learning is a field of",
            "The quick brown fox",
            "In the year 2024,",
            "Artificial intelligence has",
        ]

    def _prepare_visualization_data(
        self,
        all_metrics: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Prepare data for visualization.

        Args:
            all_metrics: Per-prompt stability metrics

        Returns:
            Dictionary with visualization data
        """
        # Collect all position metrics
        positions = []
        cv_values = []
        consistency_values = []

        for m in all_metrics:
            for pos_m in m.get("position_metrics", []):
                positions.append(pos_m.get("position", 0))
                cv_values.append(pos_m.get("coefficient_of_variation", 0))
                consistency_values.append(pos_m.get("top_token_consistency", 0))

        # CV distribution per prompt
        prompt_cvs = [m["avg_coefficient_of_variation"] for m in all_metrics]
        prompt_consistency = [m["avg_top_token_consistency"] for m in all_metrics]

        return {
            "cv_by_position": {
                "type": "scatter",
                "x": positions,
                "y": cv_values,
                "xlabel": "Token Position",
                "ylabel": "Coefficient of Variation",
                "title": "Probability Variability by Position",
            },
            "consistency_by_position": {
                "type": "scatter",
                "x": positions,
                "y": consistency_values,
                "xlabel": "Token Position",
                "ylabel": "Top-Token Consistency",
                "title": "Top-Token Consistency by Position",
            },
            "cv_distribution": {
                "type": "histogram",
                "data": prompt_cvs,
                "xlabel": "Average CV",
                "ylabel": "Count",
                "title": "Distribution of Coefficient of Variation",
            },
            "stability_scatter": {
                "type": "scatter",
                "x": prompt_cvs,
                "y": prompt_consistency,
                "xlabel": "Coefficient of Variation (lower is better)",
                "ylabel": "Top-Token Consistency (higher is better)",
                "title": "Stability Analysis: CV vs Consistency",
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
        cv = metrics.get("coefficient_of_variation", {}).get("mean", 0)
        consistency = metrics.get("top_token_consistency", {}).get("mean", 0)
        jaccard = metrics.get("jaccard_similarity", {}).get("mean", 0)

        interpretation_parts = []

        # CV interpretation
        if cv < self.CV_HEALTHY_THRESHOLD:
            interpretation_parts.append(
                f"Probability distributions are highly stable (CV={cv:.3f})."
            )
        elif cv < self.CV_WARNING_THRESHOLD:
            interpretation_parts.append(
                f"Probability distributions show moderate variability (CV={cv:.3f})."
            )
        else:
            interpretation_parts.append(
                f"Probability distributions are unstable (CV={cv:.3f}), "
                f"indicating high sensitivity to sampling randomness."
            )

        # Consistency interpretation
        if consistency > self.CONSISTENCY_HEALTHY_THRESHOLD:
            interpretation_parts.append(
                f"Top predictions are highly consistent ({consistency:.1%} agreement across samples)."
            )
        elif consistency > self.CONSISTENCY_WARNING_THRESHOLD:
            interpretation_parts.append(
                f"Top predictions show moderate consistency ({consistency:.1%} agreement)."
            )
        else:
            interpretation_parts.append(
                f"Top predictions are inconsistent ({consistency:.1%} agreement), "
                f"the model frequently changes its top choice."
            )

        # Jaccard interpretation
        interpretation_parts.append(
            f"Average Jaccard similarity of top-{self.top_k_analysis} tokens: {jaccard:.2f}."
        )

        # Position analysis if available
        pos_analysis = metrics.get("position_analysis", {})
        if pos_analysis:
            early = pos_analysis.get("early_tokens", {})
            late = pos_analysis.get("late_tokens", {})
            early_cv = early.get("avg_cv", 0)
            late_cv = late.get("avg_cv", 0)

            if late_cv > early_cv * 1.5:
                interpretation_parts.append(
                    "Stability decreases for later tokens in the generation."
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
