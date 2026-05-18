"""
Unit tests for L5 Report & Orchestration Layer.

Tests the clinic running and reporting functionality.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

from tokenpulse.clinic.runner import ClinicRunner, ClinicConfig
from tokenpulse.clinic.report import ReportGenerator
from tokenpulse.clinic.scorer import HealthScorer, ScoringWeights
from tokenpulse.clinic.data_structures import ClinicReport
from tokenpulse.checks.check_result import CheckResult, CheckConfig, HealthStatus


class TestClinicConfig:
    """Tests for ClinicConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ClinicConfig()
        assert config.preset == "quick"
        assert config.checks is None
        assert config.parallel is False
        assert config.max_workers == 4
        assert config.sample_size == 10

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ClinicConfig(
            preset="full",
            checks=["vocab", "calibration"],
            parallel=True,
            max_workers=8,
            sample_size=20,
        )
        assert config.preset == "full"
        assert config.checks == ["vocab", "calibration"]
        assert config.parallel is True
        assert config.max_workers == 8
        assert config.sample_size == 20


class TestHealthScorer:
    """Tests for HealthScorer."""

    @pytest.fixture
    def scorer(self):
        """Create a HealthScorer instance."""
        return HealthScorer()

    @pytest.fixture
    def green_results(self):
        """Create all GREEN results."""
        return [
            CheckResult(
                check_name="check1",
                check_category="blood",
                health_status=HealthStatus.GREEN,
                health_score=90.0,
            ),
            CheckResult(
                check_name="check2",
                check_category="immune",
                health_status=HealthStatus.GREEN,
                health_score=85.0,
            ),
        ]

    @pytest.fixture
    def mixed_results(self):
        """Create mixed status results."""
        return [
            CheckResult(
                check_name="check1",
                check_category="blood",
                health_status=HealthStatus.GREEN,
                health_score=90.0,
            ),
            CheckResult(
                check_name="check2",
                check_category="immune",
                health_status=HealthStatus.YELLOW,
                health_score=60.0,
            ),
        ]

    @pytest.fixture
    def red_results(self):
        """Create results with RED status."""
        return [
            CheckResult(
                check_name="check1",
                check_category="blood",
                health_status=HealthStatus.GREEN,
                health_score=90.0,
            ),
            CheckResult(
                check_name="check2",
                check_category="immune",
                health_status=HealthStatus.RED,
                health_score=30.0,
            ),
        ]

    def test_compute_overall_score_all_green(self, scorer, green_results):
        """Test score computation with all GREEN results."""
        score = scorer.compute_overall_score(green_results)
        # Should be average of scores + bonus for all GREEN
        assert score > 85  # Base average + bonus

    def test_compute_overall_score_mixed(self, scorer, mixed_results):
        """Test score computation with mixed results."""
        score = scorer.compute_overall_score(mixed_results)
        # Should be average minus penalty for YELLOW
        assert 50 < score < 80

    def test_compute_overall_score_with_red(self, scorer, red_results):
        """Test score computation with RED status."""
        score = scorer.compute_overall_score(red_results)
        # Should be reduced due to RED penalty
        assert score < 70

    def test_compute_overall_score_empty(self, scorer):
        """Test score computation with empty results."""
        score = scorer.compute_overall_score([])
        assert score == 0.0

    def test_determine_overall_status_green(self, scorer, green_results):
        """Test status determination with all GREEN."""
        score = 90.0
        status = scorer.determine_overall_status(green_results, score)
        assert status == HealthStatus.GREEN

    def test_determine_overall_status_yellow(self, scorer, mixed_results):
        """Test status determination with YELLOW."""
        score = 60.0
        status = scorer.determine_overall_status(mixed_results, score)
        assert status == HealthStatus.YELLOW

    def test_determine_overall_status_red(self, scorer, red_results):
        """Test status determination with RED."""
        score = 50.0
        status = scorer.determine_overall_status(red_results, score)
        assert status == HealthStatus.RED

    def test_score_method(self, scorer, green_results):
        """Test the score method returns complete assessment."""
        assessment = scorer.score(green_results)

        assert "overall_score" in assessment
        assert "overall_status" in assessment
        assert "category_scores" in assessment
        assert "checks_passed" in assessment
        assert assessment["checks_passed"] == 2


class TestReportGenerator:
    """Tests for ReportGenerator."""

    @pytest.fixture
    def generator(self):
        """Create a ReportGenerator instance."""
        return ReportGenerator()

    @pytest.fixture
    def sample_report(self):
        """Create a sample ClinicReport."""
        return ClinicReport(
            report_id="test_report_001",
            created_at=datetime(2024, 1, 15, 10, 30, 0),
            model_info={"model_id": "gpt2-small"},
            check_preset="quick",
            checks_executed=["vocabulary_utilization", "token_stability"],
            checks_skipped=["calibration"],
            overall_health_score=75.0,
            overall_health_status=HealthStatus.GREEN,
            check_results=[
                CheckResult(
                    check_name="vocabulary_utilization",
                    check_category="blood",
                    raw_metrics={"utilization_ratio": 0.35, "type_token_ratio": 0.25},
                    health_status=HealthStatus.GREEN,
                    health_score=80.0,
                    execution_time=1.5,
                    samples_used=10,
                ),
                CheckResult(
                    check_name="token_stability",
                    check_category="blood",
                    raw_metrics={"coefficient_of_variation": {"mean": 0.12}},
                    health_status=HealthStatus.GREEN,
                    health_score=70.0,
                    execution_time=2.0,
                    samples_used=15,
                ),
            ],
            recommendations=["模型状态良好"],
            execution_time=3.5,
            total_samples=25,
        )

    def test_generate_markdown(self, generator, sample_report):
        """Test markdown report generation."""
        markdown = generator.generate_markdown(sample_report)

        assert "# TokenPulse 体检报告" in markdown
        assert "test_report_001" in markdown
        assert "gpt2-small" in markdown
        assert "GREEN" in markdown
        assert "75.0" in markdown
        # Check for display name (Vocabulary Utilization) instead of raw ID
        assert "Vocabulary Utilization" in markdown

    def test_generate_text(self, generator, sample_report):
        """Test text report generation."""
        text = generator.generate_text(sample_report)

        assert "TokenPulse 体检报告" in text
        assert "test_report_001" in text
        assert "gpt2-small" in text
        assert "GREEN" in text

    def test_generate_json(self, generator, sample_report):
        """Test JSON report generation."""
        import json
        json_str = generator.generate_json(sample_report)

        # Should be valid JSON
        data = json.loads(json_str)
        assert data["report_id"] == "test_report_001"
        assert data["model_info"]["model_id"] == "gpt2-small"

    def test_status_emoji(self, generator):
        """Test status emoji mapping."""
        assert generator.STATUS_EMOJI[HealthStatus.GREEN] == "🟢"
        assert generator.STATUS_EMOJI[HealthStatus.YELLOW] == "🟡"
        assert generator.STATUS_EMOJI[HealthStatus.RED] == "🔴"
        assert generator.STATUS_EMOJI[HealthStatus.SKIP] == "⚪"


class TestClinicReport:
    """Tests for ClinicReport."""

    @pytest.fixture
    def report(self):
        """Create a ClinicReport instance."""
        return ClinicReport(
            report_id="test_001",
            created_at=datetime.now(),
            model_info={"model_id": "test-model"},
            check_preset="quick",
            checks_executed=["check1"],
            checks_skipped=[],
            overall_health_score=80.0,
            overall_health_status=HealthStatus.GREEN,
            check_results=[
                CheckResult(
                    check_name="check1",
                    check_category="blood",
                    health_status=HealthStatus.GREEN,
                    health_score=80.0,
                )
            ],
            recommendations=[],
        )

    def test_is_healthy(self, report):
        """Test is_healthy method."""
        assert report.is_healthy() is True

        report.overall_health_status = HealthStatus.RED
        assert report.is_healthy() is False

    def test_needs_attention(self, report):
        """Test needs_attention method."""
        assert report.needs_attention() is False

        report.overall_health_status = HealthStatus.YELLOW
        assert report.needs_attention() is True

    def test_get_summary(self, report):
        """Test get_summary method."""
        summary = report.get_summary()

        assert summary["report_id"] == "test_001"
        assert summary["model"] == "test-model"
        assert summary["overall_score"] == 80.0
        assert summary["checks_passed"] == 1

    def test_to_dict(self, report):
        """Test to_dict method."""
        d = report.to_dict()

        assert d["report_id"] == "test_001"
        assert d["check_preset"] == "quick"
        assert isinstance(d["created_at"], str)
        assert isinstance(d["check_results"], list)


class TestClinicRunner:
    """Tests for ClinicRunner."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock provider."""
        provider = Mock()
        provider.model_id = "test-model"
        provider.check_capability = Mock(return_value=True)
        provider.get_vocab_size = Mock(return_value=50257)
        return provider

    @pytest.fixture
    def config(self):
        """Create a ClinicConfig."""
        return ClinicConfig(preset="quick", sample_size=5)

    @pytest.fixture
    def mock_outputs(self):
        """Create mock model outputs."""
        from tokenpulse.providers.data_structures import ModelOutput
        import torch

        outputs = []
        for i in range(3):
            output = Mock(spec=ModelOutput)
            output.tokens = list(range(100 + i * 10, 150 + i * 10))
            output.token_strings = [f"token_{j}" for j in output.tokens]
            output.generated_text = f"Generated text {i}"
            output.logits = torch.randn(10, 1000)  # Mock logits
            output.metadata = {"input_length": 2}
            outputs.append(output)
        return outputs

    def test_initialization(self, mock_provider, config):
        """Test ClinicRunner initialization."""
        runner = ClinicRunner(mock_provider, config)

        assert runner.provider == mock_provider
        assert runner.config == config
        assert runner.scorer is not None
        assert runner.report_generator is not None

    def test_get_available_checks(self, mock_provider, config):
        """Test getting available checks."""
        runner = ClinicRunner(mock_provider, config)
        available = runner.get_available_checks()

        assert "vocabulary_utilization" in available
        assert "token_stability" in available
        assert "calibration" in available

    def test_get_compatible_checks(self, mock_provider, config):
        """Test getting compatible checks."""
        runner = ClinicRunner(mock_provider, config)
        compatible = runner.get_compatible_checks()

        # All checks should be compatible with our mock
        assert len(compatible) > 0

    def test_run_with_compatible_provider(self, mock_provider, config, mock_outputs):
        """Test running checks with compatible provider."""
        mock_provider.generate = Mock(return_value=mock_outputs)

        runner = ClinicRunner(mock_provider, config)
        report = runner.run(["test prompt"])

        assert report is not None
        assert isinstance(report, ClinicReport)
        assert report.model_info["model_id"] == "test-model"
        assert report.check_preset == "quick"

    def test_run_with_incompatible_provider(self, config):
        """Test running with incompatible provider."""
        provider = Mock()
        provider.model_id = "incompatible-model"
        provider.check_capability = Mock(return_value=False)

        runner = ClinicRunner(provider, config)
        report = runner.run(["test"])

        # Should have skipped checks
        assert len(report.checks_skipped) > 0 or len(report.check_results) == 0

    def test_generate_report_markdown(self, mock_provider, config, mock_outputs):
        """Test markdown report generation."""
        mock_provider.generate = Mock(return_value=mock_outputs)

        runner = ClinicRunner(mock_provider, config)
        runner.run(["test"])

        markdown = runner.generate_report_markdown()

        assert "# TokenPulse 体检报告" in markdown

    def test_generate_report_text(self, mock_provider, config, mock_outputs):
        """Test text report generation."""
        mock_provider.generate = Mock(return_value=mock_outputs)

        runner = ClinicRunner(mock_provider, config)
        runner.run(["test"])

        text = runner.generate_report_text()

        assert "TokenPulse 体检报告" in text

    def test_generate_report_json(self, mock_provider, config, mock_outputs):
        """Test JSON report generation."""
        mock_provider.generate = Mock(return_value=mock_outputs)

        runner = ClinicRunner(mock_provider, config)
        runner.run(["test"])

        json_output = runner.generate_report_json()

        assert '"report_id"' in json_output

    def test_custom_preset(self, mock_provider, mock_outputs):
        """Test custom preset with specific checks."""
        config = ClinicConfig(
            preset="custom",
            checks=["vocabulary_utilization"],
            sample_size=3,
        )
        mock_provider.generate = Mock(return_value=mock_outputs)

        runner = ClinicRunner(mock_provider, config)
        report = runner.run(["test"])

        # Should run only the specified check
        assert len(report.checks_executed) <= 1

    def test_presets_defined(self, mock_provider, config):
        """Test that presets are defined."""
        runner = ClinicRunner(mock_provider, config)

        assert "quick" in runner.PRESETS
        assert "full" in runner.PRESETS
        assert "calibration" in runner.PRESETS


class TestIntegration:
    """Integration tests for the complete pipeline."""

    @pytest.fixture
    def mock_provider(self):
        """Create a realistic mock provider."""
        from tokenpulse.providers.data_structures import ModelOutput
        import torch

        provider = Mock()
        provider.model_id = "integration-test-model"
        provider.check_capability = Mock(return_value=True)
        provider.get_vocab_size = Mock(return_value=50257)

        # Create realistic outputs
        def generate_side_effect(prompts, **kwargs):
            outputs = []
            for prompt in prompts if isinstance(prompts, list) else [prompts]:
                output = Mock(spec=ModelOutput)
                output.tokens = list(range(100, 150))
                output.token_strings = [f"token_{i}" for i in output.tokens]
                output.generated_text = f"Response to: {prompt}"
                output.logits = torch.randn(10, 1000)
                output.metadata = {"input_length": 5}
                outputs.append(output)
            return outputs

        provider.generate = Mock(side_effect=generate_side_effect)
        return provider

    def test_full_pipeline_quick(self, mock_provider):
        """Test full pipeline with quick preset."""
        config = ClinicConfig(preset="quick", sample_size=3)
        runner = ClinicRunner(mock_provider, config)

        prompts = ["What is AI?", "How does machine learning work?"]
        report = runner.run(prompts)

        # Verify report structure
        assert report.report_id is not None
        assert report.created_at is not None
        assert report.overall_health_score >= 0
        assert report.overall_health_score <= 100

        # Verify markdown output
        markdown = runner.generate_report_markdown()
        assert len(markdown) > 100

    def test_full_pipeline_full(self, mock_provider):
        """Test full pipeline with full preset."""
        config = ClinicConfig(preset="full", sample_size=3)
        runner = ClinicRunner(mock_provider, config)

        report = runner.run(["Test prompt"])

        # Full preset should run more checks
        assert report.check_preset == "full"

    def test_report_summary_consistency(self, mock_provider):
        """Test that report summary is consistent with full report."""
        config = ClinicConfig(preset="quick", sample_size=3)
        runner = ClinicRunner(mock_provider, config)

        report = runner.run(["Test"])
        summary = report.get_summary()

        assert summary["overall_score"] == report.overall_health_score
        assert summary["model"] == report.model_info["model_id"]
        assert summary["preset"] == report.check_preset
