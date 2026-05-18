"""
Check result data structures for L4 diagnostic modules.

This module defines the standard output format for diagnostic checks.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum


class HealthStatus(Enum):
    """Health status levels for diagnostic results."""

    GREEN = "green"      # Healthy - no issues detected
    YELLOW = "yellow"    # Warning - needs attention
    RED = "red"          # Critical - requires action
    SKIP = "skip"        # Skipped - not tested


@dataclass
class CheckConfig:
    """Configuration for a diagnostic check."""

    sample_size: int = 100
    batch_size: int = 1
    seed: int = 42
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CheckResult:
    """
    Standard result format for diagnostic checks.

    This dataclass captures all relevant information from a diagnostic check,
    including raw metrics, health assessment, and visualization data.
    """

    # Basic identification
    check_name: str
    check_category: str  # blood/neural/bone/immune

    # Raw metrics
    raw_metrics: Dict[str, Any] = field(default_factory=dict)

    # Health assessment
    health_status: HealthStatus = HealthStatus.SKIP
    health_score: float = 0.0  # 0-100

    # Detailed information
    details: Dict[str, Any] = field(default_factory=dict)
    visualization_data: Dict[str, Any] = field(default_factory=dict)

    # Comparison with baseline
    baseline_name: Optional[str] = None
    baseline_deviation: Optional[float] = None

    # Execution metadata
    execution_time: float = 0.0
    samples_used: int = 0
    error_message: Optional[str] = None

    # Interpretation (human-readable explanation)
    interpretation: Optional[str] = None

    def is_healthy(self) -> bool:
        """Check if the result indicates healthy status."""
        return self.health_status == HealthStatus.GREEN

    def needs_attention(self) -> bool:
        """Check if the result needs attention."""
        return self.health_status in (HealthStatus.YELLOW, HealthStatus.RED)

    def has_error(self) -> bool:
        """Check if an error occurred during the check."""
        return self.error_message is not None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "check_name": self.check_name,
            "check_category": self.check_category,
            "raw_metrics": self.raw_metrics,
            "health_status": self.health_status.value,
            "health_score": self.health_score,
            "details": self.details,
            "visualization_data": self.visualization_data,
            "baseline_name": self.baseline_name,
            "baseline_deviation": self.baseline_deviation,
            "execution_time": self.execution_time,
            "samples_used": self.samples_used,
            "error_message": self.error_message,
            "interpretation": self.interpretation,
        }

    def get_interpretation(self) -> str:
        """Get the interpretation string, or a default message if not set."""
        if self.interpretation:
            return self.interpretation
        if self.has_error():
            return f"检查出错: {self.error_message}"
        return f"{self.check_name} 检查完成，健康分数: {self.health_score:.1f}/100"
