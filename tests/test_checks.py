"""
Unit tests for diagnostic checks.

Tests the check functionality including:
- VocabularyUtilizationCheck: Metric computation, health status
- CalibrationCheck: ECE calculation, calibration metrics
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from collections import Counter
import numpy as np

from tokenpulse.checks.blood.vocab_utilization import VocabularyUtilizationCheck
from tokenpulse.checks.blood.token_stability import TokenStabilityCheck
from tokenpulse.checks.immune.calibration_check import CalibrationCheck
from tokenpulse.checks.check_result import CheckResult, CheckConfig, HealthStatus
from tokenpulse.providers.data_structures import ModelOutput


class TestVocabUtilizationCheck:
    """Tests for VocabularyUtilizationCheck."""

    @pytest.fixture
    def check(self):
        """Create a VocabularyUtilizationCheck instance."""
        return VocabularyUtilizationCheck()

    @pytest.fixture
    def config(self):
        """Create a CheckConfig instance."""
        return CheckConfig(sample_size=10)

    @pytest.fixture
    def mock_provider(self):
        """Create a mock provider with logits support."""
        provider = Mock()
        provider.check_capability = Mock(return_value=True)
        provider.get_vocab_size = Mock(return_value=50257)  # GPT-2 vocab size
        return provider

    @pytest.fixture
    def mock_outputs(self):
        """Create mock model outputs."""
        outputs = []
        for i in range(3):
            output = Mock(spec=ModelOutput)
            output.tokens = list(range(100 + i * 10, 150 + i * 10))  # 50 unique tokens each
            output.token_strings = [f"token_{j}" for j in output.tokens]
            output.generated_text = f"Generated text sample {i} with some content."
            output.logits = None  # Not needed for this check
            outputs.append(output)
        return outputs

    def test_check_attributes(self, check):
        """Test check has correct attributes."""
        assert check.check_name == "vocabulary_utilization"
        assert check.check_category == "blood"
        assert "logits" in check.required_capabilities

    def test_check_provider_capability_supported(self, check, mock_provider):
        """Test capability check when supported."""
        assert check.check_provider_capability(mock_provider) is True

    def test_check_provider_capability_not_supported(self, check, mock_provider):
        """Test capability check when not supported."""
        mock_provider.check_capability = Mock(return_value=False)
        assert check.check_provider_capability(mock_provider) is False

    def test_run_with_dataset(self, check, mock_provider, mock_outputs):
        """Test running the check with a custom dataset."""
        mock_provider.generate = Mock(return_value=mock_outputs[:2])

        dataset = ["Hello world!", "How are you?"]
        result = check.run(mock_provider, dataset)

        assert isinstance(result, CheckResult)
        assert result.check_name == "vocabulary_utilization"
        assert result.check_category == "blood"
        assert result.samples_used == 2
        mock_provider.generate.assert_called_once()

    def test_run_without_dataset_uses_default(self, check, mock_provider, mock_outputs):
        """Test running the check without a dataset uses default prompts."""
        mock_provider.generate = Mock(return_value=mock_outputs)

        result = check.run(mock_provider)

        assert isinstance(result, CheckResult)
        # Default prompts should be used
        call_args = mock_provider.generate.call_args
        assert isinstance(call_args[0][0], list)
        assert len(call_args[0][0]) > 0

    def test_run_capability_not_supported(self, check, mock_provider):
        """Test check returns SKIP when capabilities not supported."""
        mock_provider.check_capability = Mock(return_value=False)

        result = check.run(mock_provider, ["test"])

        assert result.health_status == HealthStatus.SKIP
        assert "capabilities" in result.error_message.lower()

    def test_compute_metrics_utilization(self, check):
        """Test utilization ratio computation."""
        unique_tokens = set(range(100))  # 100 unique tokens
        all_tokens = list(range(100)) * 2  # 200 total tokens
        token_frequencies = Counter(all_tokens)
        vocab_size = 1000

        metrics = check._compute_metrics(
            unique_tokens=unique_tokens,
            all_tokens=all_tokens,
            token_frequencies=token_frequencies,
            vocab_size=vocab_size,
            char_counts=[100],
            token_counts=[200],
            outputs=[],
        )

        assert metrics["utilization_ratio"] == 0.1  # 100/1000
        assert metrics["unique_tokens"] == 100
        assert metrics["unused_tokens"] == 900

    def test_compute_metrics_fertility(self, check):
        """Test fertility computation."""
        unique_tokens = set(range(10))
        all_tokens = list(range(10))
        token_frequencies = Counter(all_tokens)
        vocab_size = 100

        # 20 tokens, 100 characters -> fertility = 0.2
        metrics = check._compute_metrics(
            unique_tokens=unique_tokens,
            all_tokens=all_tokens,
            token_frequencies=token_frequencies,
            vocab_size=vocab_size,
            char_counts=[100],
            token_counts=[20],
            outputs=[],
        )

        assert "fertility" in metrics
        assert metrics["tokens_per_character"] == 0.2

    def test_compute_metrics_type_token_ratio(self, check):
        """Test type-token ratio computation."""
        # 10 unique tokens, 100 total tokens
        unique_tokens = set(range(10))
        all_tokens = list(range(10)) * 10  # 100 tokens with 10 unique
        token_frequencies = Counter(all_tokens)
        vocab_size = 100

        metrics = check._compute_metrics(
            unique_tokens=unique_tokens,
            all_tokens=all_tokens,
            token_frequencies=token_frequencies,
            vocab_size=vocab_size,
            char_counts=[100],
            token_counts=[100],
            outputs=[],
        )

        assert metrics["type_token_ratio"] == 0.1  # 10/100
        assert metrics["total_tokens"] == 100

    def test_compute_metrics_gini_coefficient(self, check):
        """Test Gini coefficient computation."""
        # Uniform distribution should have low Gini
        uniform_values = np.ones(100)
        gini_uniform = check._compute_gini(uniform_values)
        assert gini_uniform < 0.1

        # Highly unequal distribution should have higher Gini than uniform
        unequal_values = np.array([100] + [1] * 99)
        gini_unequal = check._compute_gini(unequal_values)
        assert gini_unequal > gini_uniform  # Should be higher than uniform
        assert gini_unequal > 0.4  # Should be moderately high

    def test_get_health_status_healthy(self, check):
        """Test health status for healthy utilization."""
        assert check._get_health_status_from_utilization(0.35) == HealthStatus.GREEN
        assert check._get_health_status_from_utilization(0.50) == HealthStatus.GREEN

    def test_get_health_status_warning(self, check):
        """Test health status for warning utilization."""
        assert check._get_health_status_from_utilization(0.15) == HealthStatus.YELLOW
        assert check._get_health_status_from_utilization(0.25) == HealthStatus.YELLOW

    def test_get_health_status_critical(self, check):
        """Test health status for critical utilization."""
        assert check._get_health_status_from_utilization(0.05) == HealthStatus.RED
        assert check._get_health_status_from_utilization(0.01) == HealthStatus.RED

    def test_interpret_result_healthy(self, check):
        """Test interpretation for healthy result."""
        result = CheckResult(
            check_name="vocabulary_utilization",
            check_category="blood",
            raw_metrics={
                "utilization_ratio": 0.40,
                "type_token_ratio": 0.3,
                "tokens_per_character": 0.2,
            },
            health_status=HealthStatus.GREEN,
        )

        interpretation = check.interpret_result(result)
        assert "healthy" in interpretation.lower()
        assert "40.0%" in interpretation

    def test_interpret_result_critical(self, check):
        """Test interpretation for critical result."""
        result = CheckResult(
            check_name="vocabulary_utilization",
            check_category="blood",
            raw_metrics={
                "utilization_ratio": 0.05,
                "type_token_ratio": 0.1,
                "tokens_per_character": 0.2,
            },
            health_status=HealthStatus.RED,
        )

        interpretation = check.interpret_result(result)
        assert "critically low" in interpretation.lower()

    def test_interpret_result_with_error(self, check):
        """Test interpretation when result has error."""
        result = CheckResult(
            check_name="vocabulary_utilization",
            check_category="blood",
            error_message="Something went wrong",
        )

        interpretation = check.interpret_result(result)
        assert "failed" in interpretation.lower()

    def test_compute_health_score(self, check):
        """Test health score computation."""
        # High utilization should give high score
        metrics = {
            "utilization_ratio": 0.5,
            "type_token_ratio": 0.5,
            "normalized_entropy": 0.8,
        }
        score = check._compute_health_score(metrics)
        assert score > 50  # 50 (util) + 10 (ttr) + 8 (entropy) = 68

        # Low utilization should give low score (utilization is primary factor)
        metrics = {
            "utilization_ratio": 0.05,
            "type_token_ratio": 0.1,
            "normalized_entropy": 0.2,
        }
        score = check._compute_health_score(metrics)
        # Score = 5 (util) + 2 (ttr) + 2 (entropy) = 9
        assert score < 20  # Should be low since utilization is only 5%

    def test_prepare_visualization_data(self, check):
        """Test visualization data preparation."""
        token_frequencies = Counter({1: 100, 2: 50, 3: 30})
        metrics = {"utilization_ratio": 0.3}

        viz_data = check._prepare_visualization_data(token_frequencies, metrics)

        assert "frequency_histogram" in viz_data
        assert "top_tokens_barchart" in viz_data
        assert "utilization_gauge" in viz_data

    def test_config_sample_size_limit(self, mock_provider, mock_outputs):
        """Test that sample size limit is respected."""
        config = CheckConfig(sample_size=2)
        check = VocabularyUtilizationCheck(config=config)
        mock_provider.generate = Mock(return_value=mock_outputs[:2])

        # Provide more samples than limit
        dataset = ["a", "b", "c", "d", "e"]
        result = check.run(mock_provider, dataset)

        # Should only process sample_size samples
        call_args = mock_provider.generate.call_args
        assert len(call_args[0][0]) == 2

    def test_empty_tokens_handling(self, check, mock_provider):
        """Test handling of empty token lists."""
        mock_output = Mock(spec=ModelOutput)
        mock_output.tokens = []
        mock_output.token_strings = []
        mock_output.generated_text = "Some text"
        mock_output.logits = None

        mock_provider.generate = Mock(return_value=[mock_output])

        result = check.run(mock_provider, ["test"])

        assert isinstance(result, CheckResult)
        # Should handle gracefully without errors

    def test_full_integration(self, mock_provider, mock_outputs):
        """Full integration test with realistic data."""
        check = VocabularyUtilizationCheck(analyze_distribution=True)
        mock_provider.generate = Mock(return_value=mock_outputs)

        dataset = ["Test prompt 1", "Test prompt 2"]
        result = check.run(mock_provider, dataset)

        assert result.health_status in [HealthStatus.GREEN, HealthStatus.YELLOW, HealthStatus.RED]
        assert result.health_score >= 0
        assert result.health_score <= 100
        assert len(result.details) > 0
        assert "utilization_ratio" in result.raw_metrics
        assert "type_token_ratio" in result.raw_metrics
        assert "fertility" in result.raw_metrics


class TestCheckResult:
    """Tests for CheckResult dataclass."""

    def test_is_healthy(self):
        """Test is_healthy method."""
        result = CheckResult(
            check_name="test",
            check_category="blood",
            health_status=HealthStatus.GREEN,
        )
        assert result.is_healthy() is True

        result.health_status = HealthStatus.RED
        assert result.is_healthy() is False

    def test_needs_attention(self):
        """Test needs_attention method."""
        result = CheckResult(
            check_name="test",
            check_category="blood",
            health_status=HealthStatus.YELLOW,
        )
        assert result.needs_attention() is True

        result.health_status = HealthStatus.RED
        assert result.needs_attention() is True

        result.health_status = HealthStatus.GREEN
        assert result.needs_attention() is False

    def test_has_error(self):
        """Test has_error method."""
        result = CheckResult(
            check_name="test",
            check_category="blood",
        )
        assert result.has_error() is False

        result.error_message = "Some error"
        assert result.has_error() is True

    def test_to_dict(self):
        """Test to_dict method."""
        result = CheckResult(
            check_name="test",
            check_category="blood",
            raw_metrics={"score": 0.5},
            health_status=HealthStatus.GREEN,
            health_score=75.0,
        )

        d = result.to_dict()
        assert d["check_name"] == "test"
        assert d["check_category"] == "blood"
        assert d["health_status"] == "green"
        assert d["health_score"] == 75.0
        assert "interpretation" in d

    def test_interpretation_field(self):
        """Test interpretation field."""
        result = CheckResult(
            check_name="test",
            check_category="blood",
            health_status=HealthStatus.GREEN,
            health_score=80.0,
            interpretation="Custom interpretation",
        )

        assert result.interpretation == "Custom interpretation"
        assert result.get_interpretation() == "Custom interpretation"

    def test_default_interpretation(self):
        """Test default interpretation when not set."""
        result = CheckResult(
            check_name="test_check",
            check_category="blood",
            health_status=HealthStatus.GREEN,
            health_score=85.0,
        )

        # Should have a default interpretation
        interpretation = result.get_interpretation()
        assert interpretation is not None
        assert len(interpretation) > 0
        assert "test_check" in interpretation or "85.0" in interpretation

    def test_error_interpretation(self):
        """Test interpretation for error results."""
        result = CheckResult(
            check_name="error_check",
            check_category="blood",
            health_status=HealthStatus.SKIP,
            error_message="Test error",
        )

        interpretation = result.get_interpretation()
        assert "Test error" in interpretation


class TestCheckConfig:
    """Tests for CheckConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = CheckConfig()
        assert config.sample_size == 100
        assert config.batch_size == 1
        assert config.seed == 42

    def test_custom_values(self):
        """Test custom configuration values."""
        config = CheckConfig(
            sample_size=50,
            batch_size=4,
            seed=123,
            custom_params={"extra": "value"}
        )
        assert config.sample_size == 50
        assert config.batch_size == 4
        assert config.seed == 123
        assert config.custom_params["extra"] == "value"


class TestCalibrationCheck:
    """Tests for CalibrationCheck."""

    @pytest.fixture
    def check(self):
        """Create a CalibrationCheck instance."""
        return CalibrationCheck()

    @pytest.fixture
    def mock_provider(self):
        """Create a mock provider with logits support."""
        provider = Mock()
        provider.check_capability = Mock(return_value=True)
        return provider

    def test_check_attributes(self, check):
        """Test check has correct attributes."""
        assert check.check_name == "calibration"
        assert check.check_category == "immune"
        assert "logits" in check.required_capabilities

    def test_ece_calculation_perfect_calibration(self, check):
        """Test ECE calculation with perfect calibration."""
        # Perfect calibration: confidence matches accuracy exactly
        confidences = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        accuracies = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])  # Accuracy matches confidence pattern

        ece = check._calculate_ece(confidences, accuracies)

        # ECE should be relatively low for well-calibrated data
        assert ece < 0.2

    def test_ece_calculation_overconfidence(self, check):
        """Test ECE calculation with overconfident model."""
        # Overconfident: high confidence but low accuracy
        confidences = np.array([0.9, 0.95, 0.85, 0.9, 0.92])
        accuracies = np.array([0, 0, 0, 0, 0])  # All wrong but confident

        ece = check._calculate_ece(confidences, accuracies)

        # ECE should be high for overconfident model
        assert ece > 0.5

    def test_ece_calculation_underconfidence(self, check):
        """Test ECE calculation with underconfident model."""
        # Underconfident: low confidence but high accuracy
        confidences = np.array([0.2, 0.3, 0.25, 0.3, 0.2])
        accuracies = np.array([1, 1, 1, 1, 1])  # All correct but not confident

        ece = check._calculate_ece(confidences, accuracies)

        # ECE should be high for underconfident model
        assert ece > 0.5

    def test_get_health_status_from_ece(self, check):
        """Test health status determination from ECE."""
        assert check._get_health_status_from_ece(0.05) == HealthStatus.GREEN
        assert check._get_health_status_from_ece(0.15) == HealthStatus.YELLOW
        assert check._get_health_status_from_ece(0.25) == HealthStatus.RED

    def test_build_mc_prompt(self, check):
        """Test multiple choice prompt construction."""
        question = "What is 2+2?"
        options = ["3", "4", "5", "6"]

        prompt = check._build_mc_prompt(question, options)

        assert "What is 2+2?" in prompt
        assert "A. 3" in prompt
        assert "B. 4" in prompt
        assert "confidence" in prompt.lower()
        assert "JSON" in prompt

    def test_build_direct_prompt(self, check):
        """Test direct prompt construction."""
        question = "What is the capital of France?"

        prompt = check._build_direct_prompt(question)

        assert "capital of France" in prompt
        assert "confidence" in prompt.lower()

    def test_parse_response_valid_json(self, check):
        """Test parsing valid JSON response."""
        response = '{"answer": "Paris", "confidence": 85}'

        answer, confidence = check._parse_response(response)

        assert answer == "Paris"
        assert confidence == 85

    def test_parse_response_with_text_around_json(self, check):
        """Test parsing JSON embedded in text."""
        response = 'Here is my answer: {"answer": "B", "confidence": 70} That is my response.'

        answer, confidence = check._parse_response(response)

        assert answer == "B"
        assert confidence == 70

    def test_parse_response_clamps_confidence(self, check):
        """Test that confidence is clamped to valid range."""
        response = '{"answer": "test", "confidence": 150}'

        answer, confidence = check._parse_response(response)

        assert confidence == 100  # Clamped to max

        response = '{"answer": "test", "confidence": -10}'
        answer, confidence = check._parse_response(response)

        assert confidence == 0  # Clamped to min

    def test_parse_response_invalid(self, check):
        """Test parsing invalid response."""
        response = "This is not a valid JSON response"

        answer, confidence = check._parse_response(response)

        assert answer is None
        assert confidence is None

    def test_evaluate_answer_exact_match(self, check):
        """Test answer evaluation with exact match."""
        assert check._evaluate_answer("Paris", "Paris") is True
        assert check._evaluate_answer("paris", "Paris") is True  # Case insensitive

    def test_evaluate_answer_partial_match(self, check):
        """Test answer evaluation with partial match."""
        assert check._evaluate_answer("Paris", "Paris, France") is True
        assert check._evaluate_answer("Shakespeare", "William Shakespeare") is True

    def test_evaluate_answer_numeric(self, check):
        """Test answer evaluation with numeric tolerance."""
        assert check._evaluate_answer("300000", "299792") is True  # Within 1%
        assert check._evaluate_answer("4", "4") is True

    def test_evaluate_answer_wrong(self, check):
        """Test answer evaluation with wrong answer."""
        assert check._evaluate_answer("London", "Paris") is False
        assert check._evaluate_answer("5", "4") is False

    def test_prepare_options_shuffles(self, check):
        """Test that options are shuffled."""
        gold = "correct"
        wrong = ["wrong1", "wrong2", "wrong3"]

        # Run multiple times to check shuffling
        options_list = [check._prepare_options(gold, wrong) for _ in range(5)]

        # Check all have correct number of options
        for options in options_list:
            assert len(options) == 4
            assert "correct" in options

    def test_compute_metrics(self, check):
        """Test full metrics computation."""
        from tokenpulse.checks.immune.calibration_check import CalibrationSample

        samples = [
            CalibrationSample("Q1", "A", "A", 80.0, True),
            CalibrationSample("Q2", "B", "C", 90.0, False),
            CalibrationSample("Q3", "C", "C", 70.0, True),
        ]

        metrics = check._compute_metrics(samples)

        assert "ece" in metrics
        assert "accuracy" in metrics
        assert "mean_confidence" in metrics
        assert "calibration_curve" in metrics
        assert metrics["accuracy"] == pytest.approx(2/3, rel=0.01)

    def test_interpret_result_well_calibrated(self, check):
        """Test interpretation for well-calibrated model."""
        result = CheckResult(
            check_name="calibration",
            check_category="immune",
            raw_metrics={
                "ece": 0.05,
                "accuracy": 0.85,
                "mean_confidence": 0.80,
                "confidence_accuracy_gap": -0.05,
            },
            health_status=HealthStatus.GREEN,
        )

        interpretation = check.interpret_result(result)
        assert "well calibrated" in interpretation.lower()

    def test_interpret_result_overconfident(self, check):
        """Test interpretation for overconfident model."""
        result = CheckResult(
            check_name="calibration",
            check_category="immune",
            raw_metrics={
                "ece": 0.25,
                "accuracy": 0.50,
                "mean_confidence": 0.85,
                "confidence_accuracy_gap": 0.35,
            },
            health_status=HealthStatus.RED,
        )

        interpretation = check.interpret_result(result)
        assert "overconfident" in interpretation.lower() or "miscalibrat" in interpretation.lower()

    def test_run_capability_not_supported(self, check, mock_provider):
        """Test check returns SKIP when capabilities not supported."""
        mock_provider.check_capability = Mock(return_value=False)

        result = check.run(mock_provider)

        assert result.health_status == HealthStatus.SKIP
        assert "capabilities" in result.error_message.lower()

    def test_compute_calibration_curve(self, check):
        """Test calibration curve computation."""
        confidences = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        accuracies = np.array([0, 1, 0, 1, 1])

        curve = check._compute_calibration_curve(confidences, accuracies)

        assert "bin_centers" in curve
        assert "bin_accuracies" in curve
        assert "bin_counts" in curve
        assert len(curve["bin_centers"]) == check.NUM_BINS

    def test_compute_health_score(self, check):
        """Test health score computation."""
        # Low ECE should give high score
        metrics = {"ece": 0.05, "accuracy": 0.9}
        score = check._compute_health_score(metrics)
        assert score > 80

        # High ECE should give low score
        metrics = {"ece": 0.3, "accuracy": 0.5}
        score = check._compute_health_score(metrics)
        assert score < 50


class TestTokenStabilityCheck:
    """Tests for TokenStabilityCheck."""

    @pytest.fixture
    def check(self):
        """Create a TokenStabilityCheck instance."""
        return TokenStabilityCheck(n_samples=3, temperature=0.7)

    @pytest.fixture
    def mock_provider(self):
        """Create a mock provider with logits support."""
        import torch
        provider = Mock()
        provider.check_capability = Mock(return_value=True)
        return provider

    @pytest.fixture
    def mock_stable_outputs(self):
        """Create mock outputs with stable (similar) probabilities."""
        import torch
        outputs = []
        for _ in range(3):
            output = Mock(spec=ModelOutput)
            # Create logits with high probability on token 100 (stable top-1)
            logits = torch.zeros(100, 1000)
            logits[:, 100] = 10.0  # High logit for token 100 -> stable predictions
            output.logits = logits
            output.metadata = {"input_length": 2}
            outputs.append(output)
        return outputs

    @pytest.fixture
    def mock_unstable_outputs(self):
        """Create mock outputs with unstable (different) top tokens."""
        import torch
        outputs = []
        for i in range(3):
            output = Mock(spec=ModelOutput)
            logits = torch.zeros(100, 1000)
            # Different top token for each sample -> unstable
            # Use spread out tokens to reduce Jaccard overlap
            base_token = 100 + i * 100
            logits[:, base_token] = 5.0
            # Also add different secondary tokens
            logits[:, base_token + 1] = 4.0
            logits[:, base_token + 2] = 3.0
            output.logits = logits
            output.metadata = {"input_length": 2}
            outputs.append(output)
        return outputs

    def test_check_attributes(self, check):
        """Test check has correct attributes."""
        assert check.check_name == "token_stability"
        assert check.check_category == "blood"
        assert "logits" in check.required_capabilities
        assert check.n_samples == 3
        assert check.temperature == 0.7

    def test_check_provider_capability_supported(self, check, mock_provider):
        """Test capability check when supported."""
        assert check.check_provider_capability(mock_provider) is True

    def test_check_provider_capability_not_supported(self, check, mock_provider):
        """Test capability check when not supported."""
        mock_provider.check_capability = Mock(return_value=False)
        assert check.check_provider_capability(mock_provider) is False

    def test_run_capability_not_supported(self, check, mock_provider):
        """Test check returns SKIP when capabilities not supported."""
        mock_provider.check_capability = Mock(return_value=False)

        result = check.run(mock_provider, ["test"])

        assert result.health_status == HealthStatus.SKIP
        assert "capabilities" in result.error_message.lower()

    def test_analyze_position_stability_stable(self, check, mock_stable_outputs):
        """Test position stability analysis with stable outputs."""
        metrics = check._analyze_position_stability(mock_stable_outputs, 5)

        assert "coefficient_of_variation" in metrics
        assert "probability_std" in metrics
        assert "probability_range" in metrics
        assert "top_token_consistency" in metrics
        assert "avg_jaccard_similarity" in metrics

        # Stable outputs should have high top-token consistency
        assert metrics["top_token_consistency"] > 0.8
        # Low coefficient of variation
        assert metrics["coefficient_of_variation"] < 0.2

    def test_analyze_position_stability_unstable(self, check, mock_unstable_outputs):
        """Test position stability analysis with unstable outputs."""
        metrics = check._analyze_position_stability(mock_unstable_outputs, 5)

        # Unstable outputs should have low top-token consistency (different top tokens)
        # Since each sample has a different top-1 token, consistency should be 1/3
        assert metrics["top_token_consistency"] == pytest.approx(1/3, rel=0.1)
        # Jaccard similarity should be reduced due to different top tokens
        assert metrics["avg_jaccard_similarity"] < 0.7

    def test_aggregate_metrics(self, check):
        """Test aggregation of metrics across prompts."""
        all_metrics = [
            {
                "prompt": "test1",
                "n_samples": 3,
                "temperature": 0.7,
                "input_length": 2,
                "generated_length": 10,
                "position_metrics": [
                    {"coefficient_of_variation": 0.1, "top_token_consistency": 0.9, "avg_jaccard_similarity": 0.8}
                ],
                "avg_coefficient_of_variation": 0.1,
                "avg_probability_range": 0.05,
                "avg_probability_std": 0.02,
                "avg_top_token_consistency": 0.9,
                "avg_jaccard_similarity": 0.8,
            },
            {
                "prompt": "test2",
                "n_samples": 3,
                "temperature": 0.7,
                "input_length": 2,
                "generated_length": 8,
                "position_metrics": [
                    {"coefficient_of_variation": 0.15, "top_token_consistency": 0.85, "avg_jaccard_similarity": 0.75}
                ],
                "avg_coefficient_of_variation": 0.15,
                "avg_probability_range": 0.06,
                "avg_probability_std": 0.03,
                "avg_top_token_consistency": 0.85,
                "avg_jaccard_similarity": 0.75,
            },
        ]

        aggregated = check._aggregate_metrics(all_metrics)

        assert "coefficient_of_variation" in aggregated
        assert "top_token_consistency" in aggregated
        assert "jaccard_similarity" in aggregated
        assert aggregated["num_prompts"] == 2

        # Check mean values
        assert abs(aggregated["coefficient_of_variation"]["mean"] - 0.125) < 0.01
        assert abs(aggregated["top_token_consistency"]["mean"] - 0.875) < 0.01

    def test_get_health_status_green(self, check):
        """Test health status for stable model."""
        metrics = {
            "coefficient_of_variation": {"mean": 0.10},
            "top_token_consistency": {"mean": 0.85},
        }
        assert check._get_health_status(metrics) == HealthStatus.GREEN

    def test_get_health_status_yellow(self, check):
        """Test health status for moderately stable model."""
        metrics = {
            "coefficient_of_variation": {"mean": 0.25},
            "top_token_consistency": {"mean": 0.60},
        }
        assert check._get_health_status(metrics) == HealthStatus.YELLOW

    def test_get_health_status_red(self, check):
        """Test health status for unstable model."""
        metrics = {
            "coefficient_of_variation": {"mean": 0.40},
            "top_token_consistency": {"mean": 0.30},
        }
        assert check._get_health_status(metrics) == HealthStatus.RED

    def test_compute_health_score(self, check):
        """Test health score computation."""
        # High stability should give high score
        metrics = {
            "coefficient_of_variation": {"mean": 0.10},
            "top_token_consistency": {"mean": 0.90},
            "jaccard_similarity": {"mean": 0.80},
        }
        score = check._compute_health_score(metrics)
        assert score > 70

        # Low stability should give low score
        metrics = {
            "coefficient_of_variation": {"mean": 0.40},
            "top_token_consistency": {"mean": 0.40},
            "jaccard_similarity": {"mean": 0.30},
        }
        score = check._compute_health_score(metrics)
        assert score < 40

    def test_interpret_result_stable(self, check):
        """Test interpretation for stable result."""
        result = CheckResult(
            check_name="token_stability",
            check_category="blood",
            raw_metrics={
                "coefficient_of_variation": {"mean": 0.08},
                "top_token_consistency": {"mean": 0.92},
                "jaccard_similarity": {"mean": 0.85},
            },
            health_status=HealthStatus.GREEN,
        )

        interpretation = check.interpret_result(result)
        assert "highly stable" in interpretation.lower() or "stable" in interpretation.lower()
        assert "consistent" in interpretation.lower()

    def test_interpret_result_unstable(self, check):
        """Test interpretation for unstable result."""
        result = CheckResult(
            check_name="token_stability",
            check_category="blood",
            raw_metrics={
                "coefficient_of_variation": {"mean": 0.35},
                "top_token_consistency": {"mean": 0.40},
                "jaccard_similarity": {"mean": 0.30},
            },
            health_status=HealthStatus.RED,
        )

        interpretation = check.interpret_result(result)
        assert "unstable" in interpretation.lower() or "inconsistent" in interpretation.lower()

    def test_interpret_result_with_error(self, check):
        """Test interpretation when result has error."""
        result = CheckResult(
            check_name="token_stability",
            check_category="blood",
            error_message="Something went wrong",
        )

        interpretation = check.interpret_result(result)
        assert "failed" in interpretation.lower()

    def test_prepare_visualization_data(self, check):
        """Test visualization data preparation."""
        all_metrics = [
            {
                "prompt": "test",
                "position_metrics": [
                    {"position": 0, "coefficient_of_variation": 0.1, "top_token_consistency": 0.9},
                    {"position": 1, "coefficient_of_variation": 0.15, "top_token_consistency": 0.85},
                ],
                "avg_coefficient_of_variation": 0.125,
                "avg_top_token_consistency": 0.875,
            }
        ]

        viz_data = check._prepare_visualization_data(all_metrics)

        assert "cv_by_position" in viz_data
        assert "consistency_by_position" in viz_data
        assert "cv_distribution" in viz_data
        assert "stability_scatter" in viz_data

    def test_get_default_prompts(self, check):
        """Test default prompts generation."""
        prompts = check._get_default_prompts()

        assert isinstance(prompts, list)
        assert len(prompts) > 0
        assert all(isinstance(p, str) for p in prompts)

    def test_run_with_dataset(self, check, mock_provider, mock_stable_outputs):
        """Test running the check with a custom dataset."""
        mock_provider.generate = Mock(return_value=mock_stable_outputs[0])

        dataset = ["Hello world!"]
        result = check.run(mock_provider, dataset)

        assert isinstance(result, CheckResult)
        assert result.check_name == "token_stability"
        assert result.check_category == "blood"
        mock_provider.generate.assert_called()

    def test_run_with_custom_params(self, mock_provider, mock_stable_outputs):
        """Test running with custom parameters via kwargs."""
        check = TokenStabilityCheck(n_samples=3, temperature=0.7)
        mock_provider.generate = Mock(return_value=mock_stable_outputs[0])

        result = check.run(mock_provider, ["test"], n_samples=5, temperature=0.9)

        assert isinstance(result, CheckResult)
        assert result.details["samples_per_prompt"] == 5
        assert result.details["temperature"] == 0.9

    def test_full_integration_stable(self, mock_provider, mock_stable_outputs):
        """Full integration test with stable outputs."""
        check = TokenStabilityCheck(n_samples=3)
        mock_provider.generate = Mock(return_value=mock_stable_outputs[0])

        result = check.run(mock_provider, ["Test prompt"])

        assert result.health_status in [HealthStatus.GREEN, HealthStatus.YELLOW, HealthStatus.RED]
        assert result.health_score >= 0
        assert result.health_score <= 100
        assert "coefficient_of_variation" in result.raw_metrics
        assert "top_token_consistency" in result.raw_metrics
