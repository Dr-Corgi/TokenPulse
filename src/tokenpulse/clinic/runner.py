"""
Clinic Runner - Orchestrates the health check pipeline.

The ClinicRunner is the main entry point for running model health checks.
It coordinates between:
- Model providers (L1)
- Diagnostic checks (L4)
- Health scoring (L5)
- Report generation (L5)
"""

from typing import List, Optional, Dict, Any, Type
from dataclasses import dataclass
import time
from datetime import datetime
import uuid

from tokenpulse.checks.base_check import BaseCheck
from tokenpulse.checks.check_result import CheckResult, HealthStatus
from tokenpulse.clinic.data_structures import ClinicReport
from tokenpulse.clinic.scorer import HealthScorer
from tokenpulse.clinic.report import ReportGenerator


@dataclass
class ClinicConfig:
    """
    Configuration for clinic runs.

    Attributes:
        preset: Preset name for check selection (quick/full/custom)
        checks: Custom list of check names (for custom preset)
        parallel: Whether to run checks in parallel
        max_workers: Maximum parallel workers
        sample_size: Default sample size for checks
        generate_charts: Whether to generate visualization charts
    """

    preset: str = "quick"
    checks: Optional[List[str]] = None
    parallel: bool = False  # Sequential by default for simplicity
    max_workers: int = 4
    sample_size: int = 10
    generate_charts: bool = False


class ClinicRunner:
    """
    Clinic Runner orchestrates the health check pipeline.

    Usage:
        >>> from tokenpulse import TransformerLensProvider
        >>> from tokenpulse.clinic import ClinicRunner
        >>>
        >>> provider = TransformerLensProvider("gpt2-small")
        >>> runner = ClinicRunner(provider, preset="quick")
        >>> report = runner.run(["What is AI?", "Explain quantum computing."])
        >>> print(report)
    """

    # Preset configurations
    PRESETS = {
        "quick": [
            "vocabulary_utilization",
            "token_stability",
        ],
        "full": [
            "vocabulary_utilization",
            "token_stability",
            "calibration",
        ],
        "calibration": [
            "calibration",
        ],
    }

    # Check registry - maps check names to classes
    CHECK_REGISTRY: Dict[str, Type[BaseCheck]] = {}

    def __init__(
        self,
        provider: "BaseProvider",
        config: Optional[ClinicConfig] = None,
    ):
        """
        Initialize the clinic runner.

        Args:
            provider: Model provider instance
            config: Clinic configuration
        """
        self.provider = provider
        self.config = config or ClinicConfig()

        # Initialize components
        self.scorer = HealthScorer()
        self.report_generator = ReportGenerator()

        # Results storage
        self._results: List[CheckResult] = []
        self._report: Optional[ClinicReport] = None

        # Register available checks
        self._register_checks()

    def _register_checks(self):
        """Register all available diagnostic checks."""
        if self.CHECK_REGISTRY:
            return  # Already registered

        try:
            from tokenpulse.checks.blood.vocab_utilization import VocabularyUtilizationCheck
            from tokenpulse.checks.blood.token_stability import TokenStabilityCheck
            from tokenpulse.checks.immune.calibration_check import CalibrationCheck

            self.CHECK_REGISTRY["vocabulary_utilization"] = VocabularyUtilizationCheck
            self.CHECK_REGISTRY["token_stability"] = TokenStabilityCheck
            self.CHECK_REGISTRY["calibration"] = CalibrationCheck
        except ImportError as e:
            raise ImportError(f"Failed to import checks: {e}")

    def get_available_checks(self) -> List[str]:
        """Get list of all available check names."""
        return list(self.CHECK_REGISTRY.keys())

    def get_compatible_checks(self) -> List[str]:
        """
        Get checks compatible with the current provider.

        Returns:
            List of check names that can run with this provider
        """
        compatible = []
        for name, check_class in self.CHECK_REGISTRY.items():
            # Create temporary instance to check capabilities
            check = check_class()
            if check.check_provider_capability(self.provider):
                compatible.append(name)
        return compatible

    def _get_checks_to_run(self) -> tuple[List[str], List[str]]:
        """
        Determine which checks to run based on config.

        Returns:
            Tuple of (checks_to_run, checks_to_skip)
        """
        # Determine requested checks
        if self.config.preset == "custom" and self.config.checks:
            requested = self.config.checks
        else:
            requested = self.PRESETS.get(self.config.preset, self.PRESETS["quick"])

        # Check compatibility
        compatible = self.get_compatible_checks()

        checks_to_run = [c for c in requested if c in compatible]
        checks_to_skip = [c for c in requested if c not in compatible]

        return checks_to_run, checks_to_skip

    def run(
        self,
        dataset: Optional[List[str]] = None,
        **kwargs
    ) -> ClinicReport:
        """
        Execute the health check pipeline.

        Args:
            dataset: Optional list of prompts for evaluation
            **kwargs: Additional parameters passed to checks

        Returns:
            ClinicReport with all results
        """
        start_time = time.time()

        # Determine checks to run
        checks_to_run, checks_to_skip = self._get_checks_to_run()

        if not checks_to_run:
            # No compatible checks
            report = ClinicReport(
                report_id=self._generate_report_id(),
                created_at=datetime.now(),
                model_info={"model_id": getattr(self.provider, "model_id", "unknown")},
                check_preset=self.config.preset,
                checks_executed=[],
                checks_skipped=checks_to_skip,
                overall_health_score=0.0,
                overall_health_status=HealthStatus.SKIP,
                check_results=[],
                recommendations=["No compatible checks available for this provider."],
                execution_time=time.time() - start_time,
            )
            self._report = report
            return report

        # Execute checks sequentially
        results: List[CheckResult] = []

        for check_name in checks_to_run:
            check_class = self.CHECK_REGISTRY.get(check_name)
            if not check_class:
                continue

            # Create check instance with config
            from tokenpulse.checks.check_result import CheckConfig
            check_config = CheckConfig(sample_size=self.config.sample_size)
            check = check_class(config=check_config)

            try:
                result = check.run(self.provider, dataset, **kwargs)
                results.append(result)
            except Exception as e:
                # Create error result
                error_result = CheckResult(
                    check_name=check_name,
                    check_category=getattr(check, "check_category", "unknown"),
                    health_status=HealthStatus.SKIP,
                    error_message=str(e),
                )
                results.append(error_result)

        self._results = results

        # Compute overall score
        scoring = self.scorer.score(results)

        # Generate recommendations
        recommendations = self._generate_recommendations(results)

        # Create report
        report = ClinicReport(
            report_id=self._generate_report_id(),
            created_at=datetime.now(),
            model_info={"model_id": getattr(self.provider, "model_id", "unknown")},
            check_preset=self.config.preset,
            checks_executed=[r.check_name for r in results if r.health_status != HealthStatus.SKIP],
            checks_skipped=checks_to_skip,
            overall_health_score=scoring["overall_score"],
            overall_health_status=scoring["overall_status"],
            check_results=results,
            recommendations=recommendations,
            execution_time=time.time() - start_time,
            total_samples=sum(r.samples_used for r in results),
        )

        self._report = report
        return report

    def _generate_report_id(self) -> str:
        """Generate a unique report ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:6]
        return f"clinic_{timestamp}_{unique_id}"

    def _generate_recommendations(self, results: List[CheckResult]) -> List[str]:
        """
        Generate recommendations based on check results.

        Args:
            results: List of check results

        Returns:
            List of recommendation strings
        """
        recommendations = []

        for result in results:
            if result.health_status == HealthStatus.RED:
                if result.check_name == "vocabulary_utilization":
                    recommendations.append(
                        "词表利用率过低，建议检查训练数据多样性和分词器配置。"
                    )
                elif result.check_name == "token_stability":
                    recommendations.append(
                        "Token概率稳定性差，模型可能存在输出不一致问题，建议检查模型权重或推理参数。"
                    )
                elif result.check_name == "calibration":
                    recommendations.append(
                        "模型置信度校准较差，建议进行校准微调或使用温度缩放。"
                    )

            elif result.health_status == HealthStatus.YELLOW:
                if result.check_name == "vocabulary_utilization":
                    recommendations.append(
                        "词表利用率略低，建议关注输出多样性。"
                    )
                elif result.check_name == "calibration":
                    recommendations.append(
                        "模型置信度与实际准确率存在偏差，建议监控校准情况。"
                    )

        # Deduplicate
        return list(dict.fromkeys(recommendations))

    def get_result(self) -> Optional[ClinicReport]:
        """Get the most recent clinic report."""
        return self._report

    def generate_report_markdown(self) -> str:
        """Generate markdown report from the latest run."""
        if self._report is None:
            return "No report available. Run the clinic first."
        return self.report_generator.generate_markdown(self._report)

    def generate_report_text(self) -> str:
        """Generate plain text report from the latest run."""
        if self._report is None:
            return "No report available. Run the clinic first."
        return self.report_generator.generate_text(self._report)

    def generate_report_json(self) -> str:
        """Generate JSON report from the latest run."""
        if self._report is None:
            return "{}"
        return self.report_generator.generate_json(self._report)
