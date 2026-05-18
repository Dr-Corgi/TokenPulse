"""
Calibration Check - Expected Calibration Error (ECE) analysis.

This check evaluates how well a model's confidence scores align with its actual accuracy.
Lower ECE indicates better calibration (confidence matches accuracy).

Adapted from llms-calibration reference implementation.
Reference: "Mind the Confidence Gap: Overconfidence, Calibration, and Distractor Effects
in Large Language Models" (TMLR)
"""

from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import time
import re
import json
import random

import numpy as np

from tokenpulse.checks.base_check import BaseCheck
from tokenpulse.checks.check_result import CheckResult, CheckConfig, HealthStatus


@dataclass
class CalibrationSample:
    """A single calibration evaluation sample."""

    question: str
    gold_answer: str
    predicted_answer: str
    confidence: float  # 0-100 scale
    is_correct: bool
    options: Optional[List[str]] = None
    raw_response: Optional[str] = None


class CalibrationCheck(BaseCheck):
    """
    Calibration Check - Evaluates model confidence calibration.

    This check measures:
    1. Expected Calibration Error (ECE): Weighted difference between confidence and accuracy
    2. Accuracy: Overall correctness rate
    3. Confidence Distribution: How confidence scores are distributed
    4. Overconfidence/Underconfidence: Systematic bias in confidence

    The check uses Multiple Choice style prompts to elicit confidence scores,
    following the approach from llms-calibration.

    Health Assessment:
    - GREEN: ECE < 0.1 (well calibrated)
    - YELLOW: ECE 0.1-0.2 (moderate miscalibration)
    - RED: ECE > 0.2 (severe miscalibration)

    Example:
        >>> from tokenpulse import TransformerLensProvider
        >>> from tokenpulse.checks.immune import CalibrationCheck
        >>>
        >>> provider = TransformerLensProvider("gpt2-small")
        >>> check = CalibrationCheck()
        >>> result = check.run(provider)
        >>> print(result.raw_metrics["ece"])
        0.15
    """

    check_name = "calibration"
    check_category = "immune"
    required_capabilities = ["logits"]

    # Health thresholds for ECE
    ECE_HEALTHY_THRESHOLD = 0.10  # ECE < 0.1 is well calibrated
    ECE_WARNING_THRESHOLD = 0.20  # ECE < 0.2 is moderate miscalibration

    # Number of bins for ECE calculation
    NUM_BINS = 10

    def __init__(
        self,
        config: Optional[CheckConfig] = None,
        use_distractors: bool = True,
        num_options: int = 4,
        temperature: float = 0.0,
    ):
        """
        Initialize the calibration check.

        Args:
            config: Check configuration
            use_distractors: Whether to use multiple choice format with distractors
            num_options: Number of options for multiple choice (default: 4)
            temperature: Temperature for generation (0.0 for deterministic)
        """
        super().__init__(config)
        self.use_distractors = use_distractors
        self.num_options = num_options
        self.temperature = temperature

    def run(
        self,
        provider: "BaseProvider",
        dataset: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> CheckResult:
        """
        Run the calibration check.

        Args:
            provider: Model provider instance
            dataset: Optional list of calibration samples. Each sample should be a dict with:
                     - "question": The question to ask
                     - "answer": The correct answer
                     - "wrong_answers": (optional) List of wrong answers for MC format
            **kwargs: Additional parameters

        Returns:
            CheckResult with calibration metrics including ECE
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

        # Use default dataset if none provided
        if dataset is None:
            dataset = self._get_default_dataset()

        # Limit sample size
        dataset = dataset[:self.config.sample_size]

        try:
            # Process each question and collect confidence/accuracy pairs
            samples: List[CalibrationSample] = []

            for item in dataset:
                sample = self._process_question(provider, item)
                if sample is not None:
                    samples.append(sample)

            if not samples:
                return CheckResult(
                    check_name=self.check_name,
                    check_category=self.check_category,
                    health_status=HealthStatus.SKIP,
                    error_message="No valid samples processed",
                    execution_time=time.time() - start_time,
                )

            # Calculate calibration metrics
            metrics = self._compute_metrics(samples)

            # Determine health status
            ece = metrics["ece"]
            health_status = self._get_health_status_from_ece(ece)
            health_score = self._compute_health_score(metrics)

            # Build result
            result = CheckResult(
                check_name=self.check_name,
                check_category=self.check_category,
                raw_metrics=metrics,
                health_status=health_status,
                health_score=health_score,
                details={
                    "samples_evaluated": len(samples),
                    "use_distractors": self.use_distractors,
                    "num_options": self.num_options,
                    "accuracy": metrics["accuracy"],
                    "mean_confidence": metrics["mean_confidence"],
                },
                visualization_data=self._prepare_visualization_data(samples, metrics),
                execution_time=time.time() - start_time,
                samples_used=len(samples),
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

    def _process_question(
        self,
        provider: "BaseProvider",
        item: Dict[str, Any]
    ) -> Optional[CalibrationSample]:
        """
        Process a single question and extract confidence/answer.

        Args:
            provider: Model provider
            item: Question item with "question", "answer", and optional "wrong_answers"

        Returns:
            CalibrationSample or None if processing failed
        """
        question = item.get("question", "")
        gold_answer = item.get("answer", "")
        wrong_answers = item.get("wrong_answers", [])

        if not question or not gold_answer:
            return None

        # Prepare options for multiple choice
        if self.use_distractors and wrong_answers:
            options = self._prepare_options(gold_answer, wrong_answers)
            prompt = self._build_mc_prompt(question, options)
        else:
            options = None
            prompt = self._build_direct_prompt(question)

        try:
            # Generate response
            output = provider.generate(
                prompt,
                return_logits=False,
                return_hidden_states=False,
                return_attention_weights=False,
                temperature=self.temperature,
                max_new_tokens=100,
            )

            # Handle single output vs list
            if isinstance(output, list):
                output = output[0]

            response_text = output.generated_text

            # Parse confidence and answer from response
            predicted_answer, confidence = self._parse_response(response_text)

            if predicted_answer is None or confidence is None:
                return None

            # Evaluate correctness
            is_correct = self._evaluate_answer(predicted_answer, gold_answer)

            return CalibrationSample(
                question=question,
                gold_answer=gold_answer,
                predicted_answer=predicted_answer,
                confidence=confidence,
                is_correct=is_correct,
                options=options,
                raw_response=response_text,
            )

        except Exception:
            return None

    def _prepare_options(
        self,
        gold_answer: str,
        wrong_answers: List[str]
    ) -> List[str]:
        """
        Prepare shuffled options for multiple choice.

        Args:
            gold_answer: The correct answer
            wrong_answers: List of incorrect answers

        Returns:
            Shuffled list of options
        """
        options = [gold_answer] + wrong_answers[:self.num_options - 1]
        random.seed(self.config.seed)
        random.shuffle(options)
        return options

    def _build_mc_prompt(self, question: str, options: List[str]) -> str:
        """
        Build a multiple choice prompt for confidence elicitation.

        Args:
            question: The question to ask
            options: List of answer options

        Returns:
            Formatted prompt string
        """
        options_text = "\n".join(
            f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)
        )

        prompt = f"""Answer the following question and provide your confidence level.

Question: {question}

Options:
{options_text}

Instructions:
1. Select the best answer from the options above (A, B, C, or D)
2. Provide your confidence level (0-100) where:
   - 0-25: Low confidence (guessing)
   - 26-75: Moderate confidence
   - 76-100: High confidence (very sure)

Respond in JSON format:
{{"answer": "X", "confidence": N}}

Where X is the letter (A/B/C/D) and N is your confidence (0-100)."""
        return prompt

    def _build_direct_prompt(self, question: str) -> str:
        """
        Build a direct prompt for confidence elicitation without options.

        Args:
            question: The question to ask

        Returns:
            Formatted prompt string
        """
        prompt = f"""Answer the following question and provide your confidence level.

Question: {question}

Instructions:
1. Provide a concise answer to the question
2. Provide your confidence level (0-100) where:
   - 0-25: Low confidence (guessing)
   - 26-75: Moderate confidence
   - 76-100: High confidence (very sure)

Respond in JSON format:
{{"answer": "your answer here", "confidence": N}}

Where N is your confidence (0-100)."""
        return prompt

    def _parse_response(
        self,
        response: str
    ) -> Tuple[Optional[str], Optional[float]]:
        """
        Parse answer and confidence from model response.

        Args:
            response: Raw model response text

        Returns:
            Tuple of (answer, confidence) or (None, None) if parsing failed
        """
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)

                answer = data.get("answer", "")
                confidence = data.get("confidence")

                if confidence is not None:
                    confidence = float(confidence)
                    # Clamp to valid range
                    confidence = max(0, min(100, confidence))

                return answer, confidence

            # Fallback: try to extract confidence number
            confidence_match = re.search(r'confidence[:\s]+(\d+(?:\.\d+)?)', response, re.IGNORECASE)
            if confidence_match:
                confidence = float(confidence_match.group(1))
                confidence = max(0, min(100, confidence))

                # Try to extract answer
                answer_match = re.search(r'answer[:\s]+["\']?([^"\',\n]+)["\']?', response, re.IGNORECASE)
                answer = answer_match.group(1).strip() if answer_match else response[:50]

                return answer, confidence

            return None, None

        except (json.JSONDecodeError, ValueError, AttributeError):
            return None, None

    def _evaluate_answer(self, predicted: str, gold: str) -> bool:
        """
        Evaluate if the predicted answer matches the gold answer.

        Args:
            predicted: Predicted answer (may be a letter like "A" for MC)
            gold: Correct answer

        Returns:
            True if answers match
        """
        # Normalize both answers
        pred_normalized = predicted.strip().lower()
        gold_normalized = gold.strip().lower()

        # Direct match
        if pred_normalized == gold_normalized:
            return True

        # Check if predicted is a letter (A, B, C, D) and matches first letter of gold
        if len(pred_normalized) == 1 and pred_normalized in "abcd":
            # For MC format, check if the option letter corresponds to gold
            return gold_normalized.startswith(pred_normalized)

        # Partial match (answer contained in gold or vice versa)
        if pred_normalized in gold_normalized or gold_normalized in pred_normalized:
            return True

        # Numeric tolerance for numbers
        try:
            pred_num = float(re.search(r'[-+]?\d*\.?\d+', pred_normalized).group())
            gold_num = float(re.search(r'[-+]?\d*\.?\d+', gold_normalized).group())
            # Allow 1% tolerance
            return abs(pred_num - gold_num) < 0.01 * max(abs(pred_num), abs(gold_num), 1)
        except (AttributeError, ValueError, ZeroDivisionError):
            pass

        return False

    def _compute_metrics(self, samples: List[CalibrationSample]) -> Dict[str, Any]:
        """
        Compute all calibration metrics.

        Args:
            samples: List of calibration samples

        Returns:
            Dictionary of computed metrics
        """
        confidences = np.array([s.confidence / 100.0 for s in samples])  # Convert to 0-1 scale
        accuracies = np.array([1.0 if s.is_correct else 0.0 for s in samples])

        metrics = {}

        # 1. Expected Calibration Error (ECE)
        metrics["ece"] = self._calculate_ece(confidences, accuracies)

        # 2. Accuracy
        metrics["accuracy"] = float(np.mean(accuracies))

        # 3. Mean Confidence
        metrics["mean_confidence"] = float(np.mean(confidences))

        # 4. Confidence-Accuracy Gap (overconfidence if positive)
        metrics["confidence_accuracy_gap"] = metrics["mean_confidence"] - metrics["accuracy"]

        # 5. Calibration curve data (for reliability diagram)
        metrics["calibration_curve"] = self._compute_calibration_curve(confidences, accuracies)

        # 6. Confidence distribution
        metrics["confidence_distribution"] = {
            "mean": float(np.mean(confidences)),
            "std": float(np.std(confidences)),
            "median": float(np.median(confidences)),
            "min": float(np.min(confidences)),
            "max": float(np.max(confidences)),
        }

        # 7. Accuracy by confidence bin
        metrics["accuracy_by_bin"] = self._compute_accuracy_by_bin(confidences, accuracies)

        # 8. Overconfidence metrics
        high_conf_mask = confidences >= 0.75
        if np.sum(high_conf_mask) > 0:
            metrics["high_confidence_accuracy"] = float(np.mean(accuracies[high_conf_mask]))
        else:
            metrics["high_confidence_accuracy"] = None

        low_conf_mask = confidences < 0.25
        if np.sum(low_conf_mask) > 0:
            metrics["low_confidence_accuracy"] = float(np.mean(accuracies[low_conf_mask]))
        else:
            metrics["low_confidence_accuracy"] = None

        return metrics

    def _calculate_ece(
        self,
        confidences: np.ndarray,
        accuracies: np.ndarray
    ) -> float:
        """
        Calculate Expected Calibration Error (ECE).

        ECE = sum(|acc(conf) - conf| * n_bin / n_total) across all bins

        Args:
            confidences: Array of confidence scores (0-1 scale)
            accuracies: Array of binary correctness (1 for correct, 0 for incorrect)

        Returns:
            ECE value (lower is better, 0 is perfect calibration)
        """
        bin_width = 1.0 / self.NUM_BINS
        bins = np.linspace(0, 1, self.NUM_BINS + 1)

        ece = 0.0
        total_samples = len(confidences)

        for i in range(self.NUM_BINS):
            # Find samples in this bin
            if i == self.NUM_BINS - 1:
                # Include right edge for last bin
                bin_mask = (confidences >= bins[i]) & (confidences <= bins[i + 1])
            else:
                bin_mask = (confidences >= bins[i]) & (confidences < bins[i + 1])

            bin_size = np.sum(bin_mask)

            if bin_size > 0:
                bin_accuracy = np.mean(accuracies[bin_mask])
                bin_confidence = np.mean(confidences[bin_mask])
                bin_weight = bin_size / total_samples
                ece += bin_weight * abs(bin_accuracy - bin_confidence)

        return float(ece)

    def _compute_calibration_curve(
        self,
        confidences: np.ndarray,
        accuracies: np.ndarray
    ) -> Dict[str, List[float]]:
        """
        Compute calibration curve for reliability diagram.

        Args:
            confidences: Array of confidence scores (0-1 scale)
            accuracies: Array of binary correctness values

        Returns:
            Dictionary with bin_centers, bin_accuracies, bin_counts
        """
        bins = np.linspace(0, 1, self.NUM_BINS + 1)
        bin_centers = []
        bin_accuracies = []
        bin_counts = []

        for i in range(self.NUM_BINS):
            if i == self.NUM_BINS - 1:
                bin_mask = (confidences >= bins[i]) & (confidences <= bins[i + 1])
            else:
                bin_mask = (confidences >= bins[i]) & (confidences < bins[i + 1])

            bin_count = np.sum(bin_mask)
            bin_counts.append(int(bin_count))

            if bin_count > 0:
                bin_center = (bins[i] + bins[i + 1]) / 2
                bin_accuracy = np.mean(accuracies[bin_mask])
                bin_centers.append(float(bin_center))
                bin_accuracies.append(float(bin_accuracy))
            else:
                bin_centers.append(float((bins[i] + bins[i + 1]) / 2))
                bin_accuracies.append(0.0)

        return {
            "bin_centers": bin_centers,
            "bin_accuracies": bin_accuracies,
            "bin_counts": bin_counts,
        }

    def _compute_accuracy_by_bin(
        self,
        confidences: np.ndarray,
        accuracies: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute accuracy for low/medium/high confidence bins.

        Args:
            confidences: Array of confidence scores
            accuracies: Array of binary correctness values

        Returns:
            Dictionary with accuracy for each confidence level
        """
        low_mask = confidences < 0.33
        med_mask = (confidences >= 0.33) & (confidences < 0.67)
        high_mask = confidences >= 0.67

        result = {}

        if np.sum(low_mask) > 0:
            result["low_confidence"] = float(np.mean(accuracies[low_mask]))
        if np.sum(med_mask) > 0:
            result["medium_confidence"] = float(np.mean(accuracies[med_mask]))
        if np.sum(high_mask) > 0:
            result["high_confidence"] = float(np.mean(accuracies[high_mask]))

        return result

    def _get_health_status_from_ece(self, ece: float) -> HealthStatus:
        """Determine health status from ECE value."""
        if ece < self.ECE_HEALTHY_THRESHOLD:
            return HealthStatus.GREEN
        elif ece < self.ECE_WARNING_THRESHOLD:
            return HealthStatus.YELLOW
        else:
            return HealthStatus.RED

    def _compute_health_score(self, metrics: Dict[str, Any]) -> float:
        """
        Compute overall health score (0-100).

        The health score is based on ECE (primary) and accuracy.
        Lower ECE = higher score.

        Args:
            metrics: Computed metrics dictionary

        Returns:
            Health score from 0 to 100
        """
        ece = metrics.get("ece", 1.0)
        accuracy = metrics.get("accuracy", 0.0)

        # ECE-based score: 0 ECE = 100, 0.3 ECE = 0
        ece_score = max(0, 100 - (ece / 0.3) * 100)

        # Accuracy bonus (up to 20 points)
        accuracy_bonus = accuracy * 20

        # Final score
        score = min(ece_score + accuracy_bonus, 100)

        return score

    def _get_default_dataset(self) -> List[Dict[str, Any]]:
        """Get default calibration dataset with simple factual questions."""
        return [
            {
                "question": "What is the capital of France?",
                "answer": "Paris",
                "wrong_answers": ["London", "Berlin", "Madrid"]
            },
            {
                "question": "What is 2 + 2?",
                "answer": "4",
                "wrong_answers": ["3", "5", "6"]
            },
            {
                "question": "Which planet is closest to the Sun?",
                "answer": "Mercury",
                "wrong_answers": ["Venus", "Mars", "Earth"]
            },
            {
                "question": "What is the largest ocean on Earth?",
                "answer": "Pacific",
                "wrong_answers": ["Atlantic", "Indian", "Arctic"]
            },
            {
                "question": "In which year did World War II end?",
                "answer": "1945",
                "wrong_answers": ["1944", "1946", "1943"]
            },
            {
                "question": "What is the chemical symbol for gold?",
                "answer": "Au",
                "wrong_answers": ["Ag", "Fe", "Cu"]
            },
            {
                "question": "Who wrote Romeo and Juliet?",
                "answer": "Shakespeare",
                "wrong_answers": ["Dickens", "Austen", "Twain"]
            },
            {
                "question": "What is the square root of 144?",
                "answer": "12",
                "wrong_answers": ["10", "14", "11"]
            },
            {
                "question": "Which element has the atomic number 1?",
                "answer": "Hydrogen",
                "wrong_answers": ["Helium", "Oxygen", "Carbon"]
            },
            {
                "question": "What is the speed of light in km/s (approximately)?",
                "answer": "300000",
                "wrong_answers": ["150000", "450000", "100000"]
            },
            {
                "question": "What is the capital of Japan?",
                "answer": "Tokyo",
                "wrong_answers": ["Osaka", "Kyoto", "Seoul"]
            },
            {
                "question": "How many continents are there?",
                "answer": "7",
                "wrong_answers": ["5", "6", "8"]
            },
            {
                "question": "What is the freezing point of water in Celsius?",
                "answer": "0",
                "wrong_answers": ["-10", "10", "32"]
            },
            {
                "question": "Who painted the Mona Lisa?",
                "answer": "da Vinci",
                "wrong_answers": ["Picasso", "Van Gogh", "Michelangelo"]
            },
            {
                "question": "What is the largest mammal?",
                "answer": "Blue whale",
                "wrong_answers": ["Elephant", "Giraffe", "Hippopotamus"]
            },
        ]

    def _prepare_visualization_data(
        self,
        samples: List[CalibrationSample],
        metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Prepare data for visualization.

        Args:
            samples: List of calibration samples
            metrics: Computed metrics

        Returns:
            Dictionary with visualization data
        """
        curve = metrics.get("calibration_curve", {})

        return {
            "reliability_diagram": {
                "type": "calibration",
                "bin_centers": curve.get("bin_centers", []),
                "bin_accuracies": curve.get("bin_accuracies", []),
                "bin_counts": curve.get("bin_counts", []),
                "perfect_calibration": [0.0, 1.0],  # Diagonal line
                "xlabel": "Confidence",
                "ylabel": "Accuracy",
                "title": "Reliability Diagram",
            },
            "confidence_histogram": {
                "type": "histogram",
                "data": [s.confidence for s in samples],
                "xlabel": "Confidence Score",
                "ylabel": "Count",
                "title": "Confidence Score Distribution",
            },
            "ece_gauge": {
                "type": "gauge",
                "value": metrics.get("ece", 0),
                "max": 0.5,
                "title": "Expected Calibration Error (lower is better)",
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
        ece = metrics.get("ece", 1.0)
        accuracy = metrics.get("accuracy", 0.0)
        mean_conf = metrics.get("mean_confidence", 0.0)
        gap = metrics.get("confidence_accuracy_gap", 0.0)

        interpretation_parts = []

        # ECE interpretation
        if ece < self.ECE_HEALTHY_THRESHOLD:
            interpretation_parts.append(
                f"The model is well calibrated (ECE = {ece:.3f}). "
                f"Confidence scores accurately reflect actual accuracy."
            )
        elif ece < self.ECE_WARNING_THRESHOLD:
            interpretation_parts.append(
                f"The model shows moderate miscalibration (ECE = {ece:.3f}). "
                f"Confidence scores somewhat misalign with accuracy."
            )
        else:
            interpretation_parts.append(
                f"The model is poorly calibrated (ECE = {ece:.3f}). "
                f"Confidence scores significantly deviate from accuracy."
            )

        # Overconfidence/underconfidence
        if gap > 0.1:
            interpretation_parts.append(
                f"The model is overconfident: mean confidence ({mean_conf:.1%}) "
                f"exceeds accuracy ({accuracy:.1%}) by {gap:.1%}."
            )
        elif gap < -0.1:
            interpretation_parts.append(
                f"The model is underconfident: mean confidence ({mean_conf:.1%}) "
                f"is below accuracy ({accuracy:.1%}) by {abs(gap):.1%}."
            )
        else:
            interpretation_parts.append(
                f"Mean confidence ({mean_conf:.1%}) aligns well with "
                f"accuracy ({accuracy:.1%})."
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
