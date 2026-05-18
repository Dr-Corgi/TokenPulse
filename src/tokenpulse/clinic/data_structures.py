"""
Data structures for L5 Report & Orchestration Layer.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

from tokenpulse.checks.check_result import CheckResult, HealthStatus


@dataclass
class ClinicReport:
    """
    Complete health check report.

    This is the final output of a clinic run, containing all check results
    and overall health assessment.
    """

    # Basic information
    report_id: str
    created_at: datetime
    model_info: Dict[str, Any]

    # Check configuration
    check_preset: str  # quick/full/custom
    checks_executed: List[str]
    checks_skipped: List[str]

    # Overall assessment
    overall_health_score: float
    overall_health_status: HealthStatus

    # Individual results
    check_results: List[CheckResult]

    # Recommendations
    recommendations: List[str] = field(default_factory=list)

    # Visualization (optional)
    charts_base64: Optional[Dict[str, str]] = None

    # Metadata
    execution_time: float = 0.0
    total_samples: int = 0

    def is_healthy(self) -> bool:
        """Check if overall health status is GREEN."""
        return self.overall_health_status == HealthStatus.GREEN

    def needs_attention(self) -> bool:
        """Check if any checks need attention."""
        return self.overall_health_status in (HealthStatus.YELLOW, HealthStatus.RED)

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the report."""
        return {
            "report_id": self.report_id,
            "model": self.model_info.get("model_id", "unknown"),
            "preset": self.check_preset,
            "overall_score": self.overall_health_score,
            "overall_status": self.overall_health_status.value,
            "checks_passed": sum(1 for r in self.check_results if r.is_healthy()),
            "checks_warning": sum(1 for r in self.check_results if r.needs_attention() and not r.has_error()),
            "checks_failed": sum(1 for r in self.check_results if r.has_error()),
            "execution_time": self.execution_time,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "report_id": self.report_id,
            "created_at": self.created_at.isoformat(),
            "model_info": self.model_info,
            "check_preset": self.check_preset,
            "checks_executed": self.checks_executed,
            "checks_skipped": self.checks_skipped,
            "overall_health_score": self.overall_health_score,
            "overall_health_status": self.overall_health_status.value,
            "check_results": [r.to_dict() for r in self.check_results],
            "recommendations": self.recommendations,
            "execution_time": self.execution_time,
            "total_samples": self.total_samples,
        }
