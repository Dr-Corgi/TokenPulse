"""
Unit tests for Calibration Calculator (L3 Layer).

Tests cover:
- Expected Calibration Error (ECE)
- Brier Score
- Reliability Diagram data
- Adaptive Calibration Error (ACE)
- Static Calibration Error (SCE)
"""

import pytest
import numpy as np


class TestExpectedCalibrationError:
    """Tests for ECE calculation."""

    def test_ece_perfect_calibration(self):
        """Perfect calibration should have low ECE."""
        from tokenpulse.core.calibration import CalibrationCalculator

        # Perfect calibration: confidence matches accuracy
        confidences = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
        accuracies = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])  # Accuracy matches confidence pattern

        result = CalibrationCalculator.expected_calibration_error(confidences, accuracies, n_bins=10)

        # Should have relatively low ECE (not perfect due to bin discretization)
        assert result.ece < 0.3

    def test_ece_worst_calibration(self):
        """Worst calibration: always confident but always wrong."""
        from tokenpulse.core.calibration import CalibrationCalculator

        confidences = np.array([1.0, 1.0, 1.0, 1.0, 1.0])  # Always 100% confident
        accuracies = np.array([0, 0, 0, 0, 0])  # Always wrong

        result = CalibrationCalculator.expected_calibration_error(confidences, accuracies, n_bins=5)

        # ECE should be high (close to 1.0)
        assert result.ece > 0.8

    def test_ece_empty_input(self):
        """Empty input should raise error."""
        from tokenpulse.core.calibration import CalibrationCalculator

        with pytest.raises(ValueError):
            CalibrationCalculator.expected_calibration_error(np.array([]), np.array([]))

    def test_ece_shape_mismatch(self):
        """Shape mismatch should raise error."""
        from tokenpulse.core.calibration import CalibrationCalculator

        confidences = np.array([0.5, 0.6, 0.7])
        accuracies = np.array([1, 0])

        with pytest.raises(ValueError):
            CalibrationCalculator.expected_calibration_error(confidences, accuracies)

    def test_ece_output_structure(self):
        """Test ECE output structure."""
        from tokenpulse.core.calibration import CalibrationCalculator

        confidences = np.array([0.9, 0.8, 0.7, 0.6])
        accuracies = np.array([1, 1, 0, 0])

        result = CalibrationCalculator.expected_calibration_error(confidences, accuracies, n_bins=4)

        assert hasattr(result, 'ece')
        assert hasattr(result, 'bin_accuracies')
        assert hasattr(result, 'bin_confidences')
        assert hasattr(result, 'bin_counts')
        assert hasattr(result, 'bin_boundaries')

        assert len(result.bin_accuracies) == 4
        assert len(result.bin_confidences) == 4
        assert len(result.bin_counts) == 4
        assert len(result.bin_boundaries) == 5  # n_bins + 1

    def test_ece_bin_statistics(self):
        """Test that bin statistics are computed correctly."""
        from tokenpulse.core.calibration import CalibrationCalculator

        # 10 samples in [0.8, 1.0] bin, all correct
        confidences = np.array([0.9, 0.85, 0.95, 0.88, 0.92, 0.87, 0.91, 0.89, 0.86, 0.93])
        accuracies = np.ones(10)

        result = CalibrationCalculator.expected_calibration_error(confidences, accuracies, n_bins=10)

        # Find the bin containing these samples
        # They should be in bin 9 (0.8-0.9) or bin 10 (0.9-1.0)
        total_samples = np.sum(result.bin_counts)
        assert total_samples == 10


class TestBrierScore:
    """Tests for Brier Score calculation."""

    def test_brier_score_perfect(self):
        """Perfect predictions should have Brier Score = 0."""
        from tokenpulse.core.calibration import CalibrationCalculator

        confidences = np.array([1.0, 1.0, 0.0, 0.0])
        accuracies = np.array([1, 1, 0, 0])

        bs = CalibrationCalculator.brier_score(confidences, accuracies)
        assert np.isclose(bs, 0.0, atol=1e-10)

    def test_brier_score_worst(self):
        """Worst predictions: confident but wrong."""
        from tokenpulse.core.calibration import CalibrationCalculator

        confidences = np.array([1.0, 1.0, 1.0, 1.0])
        accuracies = np.array([0, 0, 0, 0])

        bs = CalibrationCalculator.brier_score(confidences, accuracies)
        assert np.isclose(bs, 1.0, atol=1e-5)

    def test_brier_score_random(self):
        """Random predictions (confidence=0.5) should give BS=0.25."""
        from tokenpulse.core.calibration import CalibrationCalculator

        confidences = np.array([0.5, 0.5, 0.5, 0.5])
        accuracies = np.array([0, 1, 0, 1])  # 50% accuracy

        bs = CalibrationCalculator.brier_score(confidences, accuracies)
        # (0.5-0)^2 + (0.5-1)^2 + (0.5-0)^2 + (0.5-1)^2 = 0.25*4 = 1
        # Average = 0.25
        assert np.isclose(bs, 0.25, atol=1e-5)

    def test_brier_score_range(self):
        """Brier Score should be in [0, 1]."""
        from tokenpulse.core.calibration import CalibrationCalculator

        np.random.seed(42)
        confidences = np.random.rand(100)
        accuracies = np.random.randint(0, 2, 100)

        bs = CalibrationCalculator.brier_score(confidences, accuracies)
        assert 0.0 <= bs <= 1.0


class TestReliabilityDiagram:
    """Tests for reliability diagram data generation."""

    def test_reliability_diagram_output(self):
        """Test reliability diagram output structure."""
        from tokenpulse.core.calibration import CalibrationCalculator

        np.random.seed(42)
        confidences = np.random.rand(50)
        accuracies = np.random.randint(0, 2, 50)

        confs, accs, counts = CalibrationCalculator.reliability_diagram_data(
            confidences, accuracies, n_bins=10
        )

        assert len(confs) == 10
        assert len(accs) == 10
        assert len(counts) == 10

    def test_reliability_diagram_perfect(self):
        """Test reliability diagram for perfect calibration."""
        from tokenpulse.core.calibration import CalibrationCalculator

        # Perfect calibration
        confidences = np.array([0.9, 0.8, 0.7, 0.6])
        accuracies = np.array([1, 1, 0, 0])

        confs, accs, counts = CalibrationCalculator.reliability_diagram_data(
            confidences, accuracies, n_bins=4
        )

        # Values should be close to each other for well-calibrated regions
        # This is a basic sanity check
        assert np.all(counts >= 0)


class TestAdaptiveCalibrationError:
    """Tests for ACE calculation."""

    def test_ace_vs_ece(self):
        """ACE and ECE should be similar for uniform confidence distribution."""
        from tokenpulse.core.calibration import CalibrationCalculator

        np.random.seed(42)
        # Uniform confidence distribution
        confidences = np.linspace(0.1, 0.9, 100)
        accuracies = (confidences > 0.5).astype(float)

        ece_result = CalibrationCalculator.expected_calibration_error(confidences, accuracies)
        ace = CalibrationCalculator.adaptive_calibration_error(confidences, accuracies)

        # Both should be relatively low
        assert ece_result.ece < 0.5
        assert ace < 0.5

    def test_ace_empty(self):
        """ACE for empty input."""
        from tokenpulse.core.calibration import CalibrationCalculator

        ace = CalibrationCalculator.adaptive_calibration_error(np.array([]), np.array([]))
        assert ace == 0.0


class TestStaticCalibrationError:
    """Tests for SCE calculation."""

    def test_sce_perfect_calibration(self):
        """SCE should be low for well-calibrated predictions."""
        from tokenpulse.core.calibration import CalibrationCalculator

        confidences = np.array([0.5, 0.5, 0.5, 0.5])
        accuracies = np.array([0, 1, 0, 1])  # 50% accuracy

        sce = CalibrationCalculator.static_calibration_error(confidences, accuracies)
        assert sce < 0.1

    def test_sce_vs_ece_weighting(self):
        """SCE weights all bins equally, unlike ECE."""
        from tokenpulse.core.calibration import CalibrationCalculator

        # Most samples in one bin with large error
        confidences = np.array([0.9] * 90 + [0.1] * 10)
        accuracies = np.array([0] * 90 + [1] * 10)  # Wrong on most

        ece_result = CalibrationCalculator.expected_calibration_error(confidences, accuracies)
        sce = CalibrationCalculator.static_calibration_error(confidences, accuracies)

        # Both should be high
        assert ece_result.ece > 0.5
        assert sce > 0.5


class TestUncertaintyCalibrationError:
    """Tests for UCE calculation."""

    def test_uce_perfect_calibration(self):
        """UCE should be low when uncertainty matches error rate."""
        from tokenpulse.core.calibration import CalibrationCalculator

        # Low uncertainty -> low error, high uncertainty -> high error
        uncertainties = np.array([0.1, 0.2, 0.8, 0.9])
        errors = np.array([0, 0, 1, 1])

        uce = CalibrationCalculator.uncertainty_calibration_error(uncertainties, errors, n_bins=2)
        assert uce < 0.3  # Should be relatively low

    def test_uce_miscalibrated(self):
        """UCE should be high when uncertainty doesn't match error."""
        from tokenpulse.core.calibration import CalibrationCalculator

        # Low uncertainty but high error (miscalibrated)
        uncertainties = np.array([0.1, 0.1, 0.1, 0.1])
        errors = np.array([1, 1, 1, 1])  # Always wrong despite low uncertainty

        uce = CalibrationCalculator.uncertainty_calibration_error(uncertainties, errors)
        assert uce > 0.5  # Should be high

    def test_uce_shape_mismatch(self):
        """UCE should raise error on shape mismatch."""
        from tokenpulse.core.calibration import CalibrationCalculator

        uncertainties = np.array([0.5, 0.5, 0.5])
        errors = np.array([0, 1])

        with pytest.raises(ValueError):
            CalibrationCalculator.uncertainty_calibration_error(uncertainties, errors)


class TestPyTorchCompatibility:
    """Tests for PyTorch tensor compatibility."""

    def test_entropy_torch_tensor(self):
        """Test entropy with PyTorch tensor input."""
        try:
            import torch
            from tokenpulse.core.info_theory import InfoTheoryCalculator

            probs_torch = torch.tensor([0.25, 0.25, 0.25, 0.25])
            entropy = InfoTheoryCalculator.entropy(probs_torch)

            expected = np.log(4)
            assert np.isclose(entropy, expected, rtol=1e-5)

        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_kl_divergence_torch_tensor(self):
        """Test KL divergence with PyTorch tensor input."""
        try:
            import torch
            from tokenpulse.core.info_theory import InfoTheoryCalculator

            p = torch.tensor([0.8, 0.2])
            q = torch.tensor([0.5, 0.5])

            kl = InfoTheoryCalculator.kl_divergence(p, q)

            assert isinstance(kl, (float, torch.Tensor))
            assert kl > 0

        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_ece_torch_tensor(self):
        """Test ECE with PyTorch tensor input."""
        try:
            import torch
            from tokenpulse.core.calibration import CalibrationCalculator

            confidences = torch.tensor([0.9, 0.8, 0.7, 0.6])
            accuracies = torch.tensor([1, 1, 0, 0])

            result = CalibrationCalculator.expected_calibration_error(confidences, accuracies)

            assert result.ece >= 0

        except ImportError:
            pytest.skip("PyTorch not installed")
