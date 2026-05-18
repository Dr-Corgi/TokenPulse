"""
Health Scorer - Computes overall health score from check results.

The health scorer aggregates individual check results into an overall
health assessment using configurable weighting strategies.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from tokenpulse.checks.check_result import CheckResult, HealthStatus


@dataclass
class ScoringWeights:
    """Weights for different check categories."""

    blood: float = 1.0      # Probability distribution checks
    neural: float = 1.0     # Sensitivity checks
    bone: float = 1.0       # Representation checks
    immune: float = 1.0     # Calibration checks

    def get_weight(self, category: str) -> float:
        """Get weight for a category."""
        return getattr(self, category, 1.0)


class HealthScorer:
    """
    Health Scorer aggregates check results into overall health assessment.

    The scoring strategy:
    1. Weighted average of health scores by category
    2. Penalty for RED status checks
    3. Bonus for all GREEN status

    Overall status determination:
    - GREEN: All checks pass, score >= 70
    - YELLOW: Any YELLOW check, or score 40-70
    - RED: Any RED check, or score < 40
    """

    # Default weights for each category
    DEFAULT_WEIGHTS = ScoringWeights()

    # Score thresholds for overall status
    GREEN_THRESHOLD = 70.0
    YELLOW_THRESHOLD = 40.0

    # Penalty for RED status (percentage deduction)
    RED_PENALTY = 20.0
    YELLOW_PENALTY = 5.0

    # Bonus for all GREEN (percentage addition)
    ALL_GREEN_BONUS = 10.0

    def __init__(self, weights: Optional[ScoringWeights] = None):
        """
        Initialize the health scorer.

        Args:
            weights: Custom scoring weights for categories
        """
        self.weights = weights or self.DEFAULT_WEIGHTS

    def compute_overall_score(self, results: List[CheckResult]) -> float:
        """
        Compute overall health score from check results.

        Args:
            results: List of check results

        Returns:
            Overall health score (0-100)
        """
        if not results:
            return 0.0

        # Filter out skipped/error results for scoring
        valid_results = [r for r in results if r.health_status != HealthStatus.SKIP]

        if not valid_results:
            return 0.0

        # Calculate weighted average score
        total_weight = 0.0
        weighted_score = 0.0

        for result in valid_results:
            category = result.check_category
            weight = self.weights.get_weight(category)
            total_weight += weight
            weighted_score += result.health_score * weight

        if total_weight == 0:
            return 0.0

        base_score = weighted_score / total_weight

        # Apply penalties
        penalty = 0.0
        for result in valid_results:
            if result.health_status == HealthStatus.RED:
                penalty += self.RED_PENALTY
            elif result.health_status == HealthStatus.YELLOW:
                penalty += self.YELLOW_PENALTY

        # Cap penalty at 50% of base score
        penalty = min(penalty, base_score * 0.5)

        # Apply bonus for all GREEN
        bonus = 0.0
        if all(r.health_status == HealthStatus.GREEN for r in valid_results):
            bonus = self.ALL_GREEN_BONUS

        # Final score
        final_score = base_score - penalty + bonus

        # Clamp to 0-100
        return max(0.0, min(100.0, final_score))

    def determine_overall_status(
        self,
        results: List[CheckResult],
        overall_score: float
    ) -> HealthStatus:
        """
        Determine overall health status from results and score.

        Args:
            results: List of check results
            overall_score: Computed overall score

        Returns:
            Overall health status
        """
        # Check for any RED status (critical)
        if any(r.health_status == HealthStatus.RED for r in results):
            return HealthStatus.RED

        # Check for any YELLOW status (warning)
        if any(r.health_status == HealthStatus.YELLOW for r in results):
            return HealthStatus.YELLOW

        # All GREEN - use score-based determination
        if overall_score >= self.GREEN_THRESHOLD:
            return HealthStatus.GREEN
        elif overall_score >= self.YELLOW_THRESHOLD:
            return HealthStatus.YELLOW
        else:
            return HealthStatus.RED

    def score(self, results: List[CheckResult]) -> Dict[str, Any]:
        """
        Compute complete health assessment.

        Args:
            results: List of check results

        Returns:
            Dictionary with score, status, and breakdown
        """
        overall_score = self.compute_overall_score(results)
        overall_status = self.determine_overall_status(results, overall_score)

        # Breakdown by category
        category_scores = {}
        for category in ["blood", "neural", "bone", "immune"]:
            category_results = [r for r in results if r.check_category == category]
            if category_results:
                valid = [r for r in category_results if r.health_status != HealthStatus.SKIP]
                if valid:
                    avg_score = sum(r.health_score for r in valid) / len(valid)
                    category_scores[category] = {
                        "score": avg_score,
                        "checks": len(valid),
                        "status": self._get_category_status(valid),
                    }

        return {
            "overall_score": overall_score,
            "overall_status": overall_status,
            "category_scores": category_scores,
            "checks_passed": sum(1 for r in results if r.health_status == HealthStatus.GREEN),
            "checks_warning": sum(1 for r in results if r.health_status == HealthStatus.YELLOW),
            "checks_critical": sum(1 for r in results if r.health_status == HealthStatus.RED),
            "checks_skipped": sum(1 for r in results if r.health_status == HealthStatus.SKIP),
        }

    def _get_category_status(self, results: List[CheckResult]) -> str:
        """Get the worst status from a list of results."""
        if any(r.health_status == HealthStatus.RED for r in results):
            return "red"
        if any(r.health_status == HealthStatus.YELLOW for r in results):
            return "yellow"
        return "green"
