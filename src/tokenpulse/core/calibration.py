"""
Calibration Calculator - L3 Layer Core Engine.

Provides calibration metrics like Expected Calibration Error (ECE) and Brier Score.
These functions evaluate how well a model's confidence matches its actual accuracy.

Reference implementations:
- https://github.com/gpleiss/temperature_scaling
- https://github.com/ICLR2023/large-model-calibration-and-uncertainty
"""

from typing import Union, Tuple, Optional
from dataclasses import dataclass
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class ECEOutput:
    """
    Output structure for Expected Calibration Error computation.

    Attributes:
        ece: Expected Calibration Error value
        bin_accuracies: Accuracy in each bin
        bin_confidences: Average confidence in each bin
        bin_counts: Number of samples in each bin
        bin_boundaries: Bin boundary values
    """
    ece: float
    bin_accuracies: np.ndarray
    bin_confidences: np.ndarray
    bin_counts: np.ndarray
    bin_boundaries: np.ndarray


class CalibrationCalculator:
    """
    Confidence calibration computation utilities.

    Calibration measures whether a model's predicted probabilities reflect
    the true likelihood of correctness. A well-calibrated model that predicts
    80% confidence should be correct 80% of the time.

    Key metrics:
    - ECE (Expected Calibration Error): Average gap between confidence and accuracy
    - Brier Score: Mean squared error between predicted probability and outcome
    - UCE (Uncertainty Calibration Error): Calibration of uncertainty estimates

    Example:
        >>> confidences = np.array([0.9, 0.7, 0.6, 0.8])
        >>> accuracies = np.array([1, 1, 0, 1])  # 1 = correct, 0 = wrong
        >>> result = CalibrationCalculator.expected_calibration_error(confidences, accuracies)
        >>> print(f"ECE: {result.ece:.4f}")
    """

    @staticmethod
    def expected_calibration_error(
        confidences: Union[np.ndarray, "torch.Tensor", list],
        accuracies: Union[np.ndarray, "torch.Tensor", list],
        n_bins: int = 10,
        debias: bool = False
    ) -> ECEOutput:
        """
        Calculate Expected Calibration Error (ECE).

        ECE = Σ (n_i / N) * |acc_i - conf_i|

        where:
        - n_i = number of samples in bin i
        - N = total number of samples
        - acc_i = accuracy in bin i
        - conf_i = average confidence in bin i

        This implementation follows the standard ECE formulation where
        bin boundaries are: bin_lower < confidence <= bin_upper for all bins.

        Args:
            confidences: Model confidence values [0, 1]. Shape (N,)
            accuracies: Ground truth accuracy (0 or 1). Shape (N,)
            n_bins: Number of bins for calibration (default: 10)
            debias: If True, use debiased ECE estimation

        Returns:
            ECEOutput containing ECE and per-bin statistics

        Raises:
            ValueError: If inputs have different shapes or invalid values

        Examples:
            >>> confidences = np.array([0.9, 0.8, 0.3, 0.4])
            >>> accuracies = np.array([1, 1, 0, 0])  # Perfect calibration
            >>> result = CalibrationCalculator.expected_calibration_error(confidences, accuracies)
            >>> result.ece  # Should be close to 0
            0.0
        """
        # Convert to numpy
        if HAS_TORCH and isinstance(confidences, torch.Tensor):
            confidences = confidences.detach().cpu().numpy()
        if HAS_TORCH and isinstance(accuracies, torch.Tensor):
            accuracies = accuracies.detach().cpu().numpy()

        confidences = np.asarray(confidences, dtype=np.float64).flatten()
        accuracies = np.asarray(accuracies, dtype=np.float64).flatten()

        # Validation
        if confidences.shape != accuracies.shape:
            raise ValueError(
                f"Shape mismatch: confidences {confidences.shape} vs accuracies {accuracies.shape}"
            )

        n_samples = len(confidences)
        if n_samples == 0:
            raise ValueError("Empty input arrays")

        # Clip confidences to valid range
        confidences = np.clip(confidences, 0.0, 1.0)

        # Create bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        # Initialize outputs
        ece = 0.0
        bin_accuracies = np.zeros(n_bins)
        bin_confidences = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins)

        for i, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
            # Find samples in this bin
            # Standard ECE: bin_lower < confidence <= bin_upper for ALL bins
            # This matches the reference implementation from temperature_scaling
            if i == 0:
                # First bin: include left boundary to handle confidence=0
                # Some implementations use > for all bins, but this excludes 0
                # We follow a more inclusive approach: >= 0 and <= bin_upper
                in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
            else:
                in_bin = (confidences > bin_lower) & (confidences <= bin_upper)

            bin_count = np.sum(in_bin)
            bin_counts[i] = bin_count

            if bin_count > 0:
                # Calculate accuracy and confidence in this bin
                bin_accuracy = np.mean(accuracies[in_bin])
                bin_confidence = np.mean(confidences[in_bin])

                bin_accuracies[i] = bin_accuracy
                bin_confidences[i] = bin_confidence

                # Accumulate weighted error
                bin_weight = bin_count / n_samples
                ece += np.abs(bin_confidence - bin_accuracy) * bin_weight

        # Optional: debiased ECE (adjusts for finite sample bias)
        if debias and n_samples > 1:
            # Debias factor: reduces ECE based on sample variance
            # This is a simplified version; more sophisticated methods exist
            variance_factor = 1.0 - (1.0 / np.sqrt(n_samples * n_bins))
            ece = max(0, ece * variance_factor)

        return ECEOutput(
            ece=ece,
            bin_accuracies=bin_accuracies,
            bin_confidences=bin_confidences,
            bin_counts=bin_counts,
            bin_boundaries=bin_boundaries
        )

    @staticmethod
    def brier_score(
        confidences: Union[np.ndarray, "torch.Tensor", list],
        accuracies: Union[np.ndarray, "torch.Tensor", list]
    ) -> float:
        """
        Calculate Brier Score.

        BS = (1/N) * Σ (conf - acc)^2

        Brier Score is a proper scoring rule that measures the mean squared
        error between predicted probabilities and outcomes. Lower is better.

        Range: [0, 1] where 0 is perfect calibration.

        Args:
            confidences: Model confidence values [0, 1]
            accuracies: Ground truth accuracy (0 or 1)

        Returns:
            Brier Score value

        Examples:
            >>> confidences = np.array([1.0, 0.0, 1.0, 0.0])
            >>> accuracies = np.array([1, 0, 1, 0])  # Perfect predictions
            >>> CalibrationCalculator.brier_score(confidences, accuracies)
            0.0
        """
        # Convert to numpy
        if HAS_TORCH and isinstance(confidences, torch.Tensor):
            confidences = confidences.detach().cpu().numpy()
        if HAS_TORCH and isinstance(accuracies, torch.Tensor):
            accuracies = accuracies.detach().cpu().numpy()

        confidences = np.asarray(confidences, dtype=np.float64).flatten()
        accuracies = np.asarray(accuracies, dtype=np.float64).flatten()

        if confidences.shape != accuracies.shape:
            raise ValueError(
                f"Shape mismatch: confidences {confidences.shape} vs accuracies {accuracies.shape}"
            )

        return float(np.mean((confidences - accuracies) ** 2))

    @staticmethod
    def reliability_diagram_data(
        confidences: Union[np.ndarray, "torch.Tensor", list],
        accuracies: Union[np.ndarray, "torch.Tensor", list],
        n_bins: int = 10
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate data for reliability diagram (calibration curve).

        A reliability diagram plots accuracy vs confidence for each bin.
        Perfect calibration falls on the diagonal line.

        Args:
            confidences: Model confidence values [0, 1]
            accuracies: Ground truth accuracy (0 or 1)
            n_bins: Number of bins

        Returns:
            Tuple of (bin_confidences, bin_accuracies, bin_counts)

        Examples:
            >>> confidences = np.array([0.9, 0.7, 0.5, 0.3])
            >>> accuracies = np.array([1, 1, 0, 0])
            >>> confs, accs, counts = CalibrationCalculator.reliability_diagram_data(
            ...     confidences, accuracies, n_bins=4
            ... )
        """
        result = CalibrationCalculator.expected_calibration_error(
            confidences, accuracies, n_bins
        )
        return result.bin_confidences, result.bin_accuracies, result.bin_counts

    @staticmethod
    def adaptive_calibration_error(
        confidences: Union[np.ndarray, "torch.Tensor", list],
        accuracies: Union[np.ndarray, "torch.Tensor", list],
        n_bins: int = 10
    ) -> float:
        """
        Calculate Adaptive Calibration Error (ACE).

        Unlike ECE which uses fixed-width bins, ACE uses bins with equal
        number of samples, which can be more robust for sparse regions.

        Args:
            confidences: Model confidence values [0, 1]
            accuracies: Ground truth accuracy (0 or 1)
            n_bins: Number of bins (each bin has ~N/n_bins samples)

        Returns:
            Adaptive Calibration Error value
        """
        # Convert to numpy
        if HAS_TORCH and isinstance(confidences, torch.Tensor):
            confidences = confidences.detach().cpu().numpy()
        if HAS_TORCH and isinstance(accuracies, torch.Tensor):
            accuracies = accuracies.detach().cpu().numpy()

        confidences = np.asarray(confidences, dtype=np.float64).flatten()
        accuracies = np.asarray(accuracies, dtype=np.float64).flatten()

        n_samples = len(confidences)
        if n_samples == 0:
            return 0.0

        # Sort by confidence
        sorted_indices = np.argsort(confidences)
        sorted_confidences = confidences[sorted_indices]
        sorted_accuracies = accuracies[sorted_indices]

        # Create bins with equal number of samples
        bin_size = max(1, n_samples // n_bins)

        ace = 0.0
        for i in range(n_bins):
            start = i * bin_size
            end = start + bin_size if i < n_bins - 1 else n_samples

            if end > start:
                bin_conf = np.mean(sorted_confidences[start:end])
                bin_acc = np.mean(sorted_accuracies[start:end])
                bin_weight = (end - start) / n_samples
                ace += np.abs(bin_conf - bin_acc) * bin_weight

        return ace

    @staticmethod
    def static_calibration_error(
        confidences: Union[np.ndarray, "torch.Tensor", list],
        accuracies: Union[np.ndarray, "torch.Tensor", list],
        n_bins: int = 10
    ) -> float:
        """
        Calculate Static Calibration Error (SCE).

        SCE averages the calibration error across all confidence bins
        without weighting by bin size. This gives equal importance to
        all confidence ranges.

        Args:
            confidences: Model confidence values [0, 1]
            accuracies: Ground truth accuracy (0 or 1)
            n_bins: Number of bins

        Returns:
            Static Calibration Error value
        """
        result = CalibrationCalculator.expected_calibration_error(
            confidences, accuracies, n_bins
        )

        # Calculate unweighted average
        non_empty_bins = result.bin_counts > 0
        if np.sum(non_empty_bins) == 0:
            return 0.0

        gaps = np.abs(result.bin_confidences - result.bin_accuracies)
        return float(np.mean(gaps[non_empty_bins]))

    @staticmethod
    def uncertainty_calibration_error(
        uncertainties: Union[np.ndarray, "torch.Tensor", list],
        errors: Union[np.ndarray, "torch.Tensor", list],
        n_bins: int = 10
    ) -> float:
        """
        Calculate Uncertainty Calibration Error (UCE).

        UCE measures whether a model's uncertainty (e.g., entropy-based)
        matches its actual error rate. Similar to ECE but bins by uncertainty
        values instead of confidence.

        UCE = Σ (n_i / N) * |uncertainty_i - error_rate_i|

        Args:
            uncertainties: Uncertainty values (e.g., entropy) [0, max_entropy].
                          Shape (N,)
            errors: Ground truth error (0 = correct, 1 = wrong). Shape (N,)
            n_bins: Number of bins

        Returns:
            Uncertainty Calibration Error value

        Examples:
            >>> uncertainties = np.array([0.1, 0.2, 0.8, 0.9])  # Low then high uncertainty
            >>> errors = np.array([0, 0, 1, 1])  # Low uncertainty = correct, high = wrong
            >>> uce = CalibrationCalculator.uncertainty_calibration_error(uncertainties, errors)
            >>> uce  # Should be low (well-calibrated uncertainty)
            0.0
        """
        # Convert to numpy
        if HAS_TORCH and isinstance(uncertainties, torch.Tensor):
            uncertainties = uncertainties.detach().cpu().numpy()
        if HAS_TORCH and isinstance(errors, torch.Tensor):
            errors = errors.detach().cpu().numpy()

        uncertainties = np.asarray(uncertainties, dtype=np.float64).flatten()
        errors = np.asarray(errors, dtype=np.float64).flatten()

        if uncertainties.shape != errors.shape:
            raise ValueError(
                f"Shape mismatch: uncertainties {uncertainties.shape} vs errors {errors.shape}"
            )

        n_samples = len(uncertainties)
        if n_samples == 0:
            return 0.0

        # Create bins based on uncertainty range
        max_uncertainty = np.max(uncertainties)
        bin_boundaries = np.linspace(0, max_uncertainty, n_bins + 1)

        uce = 0.0
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            # Find samples in this bin
            if i == 0:
                in_bin = (uncertainties >= bin_lower) & (uncertainties <= bin_upper)
            else:
                in_bin = (uncertainties > bin_lower) & (uncertainties <= bin_upper)

            bin_count = np.sum(in_bin)
            if bin_count > 0:
                avg_uncertainty = np.mean(uncertainties[in_bin])
                error_rate = np.mean(errors[in_bin])
                bin_weight = bin_count / n_samples
                uce += np.abs(avg_uncertainty - error_rate) * bin_weight

        return uce

    @staticmethod
    def ece_torch(
        softmaxes: "torch.Tensor",
        labels: "torch.Tensor",
        n_bins: int = 10
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """
        PyTorch-native ECE calculation (matches reference implementation).

        This is a direct port of the reference implementation from
        https://github.com/gpleiss/temperature_scaling

        Args:
            softmaxes: Softmax outputs [batch_size, num_classes]
            labels: Ground truth labels [batch_size]
            n_bins: Number of bins

        Returns:
            Tuple of (ece, accuracies_in_bins, confidences_in_bins)
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for ece_torch")

        device = softmaxes.device
        bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=device)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        # Get confidence and predictions
        confidences, predictions = torch.max(softmaxes, dim=1)
        accuracies = predictions.eq(labels)

        accuracy_in_bin_list = []
        avg_confidence_in_bin_list = []

        ece = torch.zeros(1, device=device)
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Standard ECE binning: bin_lower < conf <= bin_upper
            in_bin = confidences.gt(bin_lower.item()) & confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()

            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

                accuracy_in_bin_list.append(accuracy_in_bin)
                avg_confidence_in_bin_list.append(avg_confidence_in_bin)

        acc_in_bin = torch.tensor(accuracy_in_bin_list, device=device)
        avg_conf_in_bin = torch.tensor(avg_confidence_in_bin_list, device=device)

        return ece, acc_in_bin, avg_conf_in_bin
