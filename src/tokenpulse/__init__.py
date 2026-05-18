"""
TokenPulse - LLM Model Diagnostic Tool

A comprehensive health check system for Large Language Models.

Usage:
    >>> from tokenpulse import TransformerLensProvider, ClinicRunner
    >>> provider = TransformerLensProvider("gpt2-small")
    >>> runner = ClinicRunner(provider, preset="quick")
    >>> report = runner.run(["What is AI?"])
    >>> print(runner.generate_report_markdown())
"""

__version__ = "0.1.0"

# Providers (L1)
from tokenpulse.providers.base import BaseProvider
from tokenpulse.providers.data_structures import ModelOutput, ModelCapabilities
from tokenpulse.providers.transformer_lens_provider import TransformerLensProvider

# Core (L3)
from tokenpulse.core.info_theory import InfoTheoryCalculator
from tokenpulse.core.calibration import CalibrationCalculator

# Checks (L4)
from tokenpulse.checks.base_check import BaseCheck
from tokenpulse.checks.check_result import CheckResult, CheckConfig, HealthStatus
from tokenpulse.checks.blood.vocab_utilization import VocabularyUtilizationCheck
from tokenpulse.checks.blood.token_stability import TokenStabilityCheck
from tokenpulse.checks.immune.calibration_check import CalibrationCheck

# Clinic (L5)
from tokenpulse.clinic.runner import ClinicRunner, ClinicConfig
from tokenpulse.clinic.report import ReportGenerator
from tokenpulse.clinic.scorer import HealthScorer
from tokenpulse.clinic.data_structures import ClinicReport

__all__ = [
    # Version
    "__version__",
    # Providers (L1)
    "BaseProvider",
    "ModelOutput",
    "ModelCapabilities",
    "TransformerLensProvider",
    # Core (L3)
    "InfoTheoryCalculator",
    "CalibrationCalculator",
    # Checks (L4)
    "BaseCheck",
    "CheckResult",
    "CheckConfig",
    "HealthStatus",
    "VocabularyUtilizationCheck",
    "TokenStabilityCheck",
    "CalibrationCheck",
    # Clinic (L5)
    "ClinicRunner",
    "ClinicConfig",
    "ReportGenerator",
    "HealthScorer",
    "ClinicReport",
]
