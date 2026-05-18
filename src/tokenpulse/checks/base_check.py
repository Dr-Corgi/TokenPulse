"""
Base class for diagnostic checks (L4 Layer).

All diagnostic checks inherit from BaseCheck and implement the run() method.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, TYPE_CHECKING

from tokenpulse.checks.check_result import CheckResult, CheckConfig

if TYPE_CHECKING:
    from tokenpulse.providers.base import BaseProvider


class BaseCheck(ABC):
    """
    Abstract base class for diagnostic checks.

    Subclasses must define:
    - check_name: Unique identifier for this check
    - check_category: One of blood/neural/bone/immune
    - required_capabilities: List of capabilities needed from provider
    - run(): The main check logic
    - interpret_result(): Human-readable interpretation
    - get_health_status(): Determine health from metrics
    """

    # Subclass must define these
    check_name: str = ""
    check_category: str = ""  # blood/neural/bone/immune
    required_capabilities: List[str] = []  # e.g., ["logits", "hidden_states"]

    def __init__(self, config: Optional[CheckConfig] = None):
        """
        Initialize the check.

        Args:
            config: Check configuration (sample size, batch size, etc.)
        """
        self.config = config or CheckConfig()
        self._result: Optional[CheckResult] = None

    @abstractmethod
    def run(
        self,
        provider: "BaseProvider",
        dataset: Optional[List[str]] = None,
        **kwargs
    ) -> CheckResult:
        """
        Execute the diagnostic check.

        Args:
            provider: Model provider instance
            dataset: Optional list of prompts for evaluation
            **kwargs: Additional check-specific parameters

        Returns:
            CheckResult containing metrics and health assessment
        """
        pass

    def check_provider_capability(self, provider: "BaseProvider") -> bool:
        """
        Check if the provider supports all required capabilities.

        Args:
            provider: Model provider to check

        Returns:
            True if all capabilities are supported
        """
        for cap in self.required_capabilities:
            if not provider.check_capability(f"supports_{cap}"):
                return False
        return True

    def get_result(self) -> Optional[CheckResult]:
        """
        Get the most recent check result.

        Returns:
            Last CheckResult or None if not run
        """
        return self._result

    @abstractmethod
    def interpret_result(self, result: CheckResult) -> str:
        """
        Generate a human-readable interpretation of the result.

        Args:
            result: Check result to interpret

        Returns:
            Human-readable string explaining the result
        """
        pass

    @abstractmethod
    def get_health_status(self, result: CheckResult) -> "HealthStatus":
        """
        Determine health status from the result.

        Args:
            result: Check result to evaluate

        Returns:
            HealthStatus enum value
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.check_name}', category='{self.check_category}')"
