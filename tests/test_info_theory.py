"""
Unit tests for Information Theory Calculator (L3 Layer).

Tests cover:
- Entropy calculation
- KL divergence calculation
- Cross-entropy calculation
- Top-p mass calculation
- Perplexity calculation
- Normalized entropy calculation
"""

import pytest
import numpy as np
import math


class TestEntropy:
    """Tests for entropy calculation."""

    def test_entropy_uniform_distribution(self):
        """Uniform distribution should have maximum entropy."""
        from tokenpulse.core.info_theory import InfoTheoryCalculator

        # Uniform distribution over 4 outcomes
        probs = np.array([0.25, 0.25, 0.25, 0.25])
        entropy = InfoTheoryCalculator.entropy(probs)

        # Expected: log(4)
        expected = np.log(4)
        assert np.isclose(entropy, expected, rtol=1e-5)

    def test_entropy_deterministic(self):
        """Deterministic distribution should have near-zero entropy."""
        from tokenpulse.core.info_theory import InfoTheoryCalculator

        probs = np.array([1.0, 0.0, 0.0, 0.0])
        entropy = InfoTheoryCalculator.entropy(probs)

        # Should be very close to zero (within numerical precision)
        assert entropy < 1e-8

    def test_entropy_binary_symmetric(self):
        """Binary symmetric distribution (0.5, 0.5) should have entropy log(2)."""
        from tokenpulse.core.info_theory import InfoTheoryCalculator

        probs = np.array([0.5, 0.5])
        entropy = InfoTheoryCalculator.entropy(probs)

        expected = np.log(2)
        assert np.isclose(entropy, expected, rtol=1e-5)

    def test_entropy_batch_computation(self):
        """Test batch computation of entropy."""
        from tokenpulse.core.info_theory import InfoTheoryCalculator

        # Batch of 3 probability distributions
        probs = np.array([
            [0.5, 0.5],      # H = log(2)
            [1.0, 0.0],      # H ≈ 0 (numerical precision)
            [0.25, 0.75],    # H = -0.25*log(0.25) - 0.75*log(0.75)
        ])

        entropies = InfoTheoryCalculator.entropy(probs, dim=1)

        assert len(entropies) == 3
        assert np.isclose(entropies[0], np.log(2), rtol=1e-5)
        assert entropies[1] < 1e-8  # Near zero
        assert entropies[2] > 0

    def test_entropy_auto_normalize(self):
        """Test that entropy auto-normalizes non-normalized distributions."""
        from tokenpulse.core.info_theory import InfoTheoryCalculator

        # Non-normalized distribution
        probs = np.array([2.0, 2.0, 2.0, 2.0])  # Sum = 8
        entropy = InfoTheoryCalculator.entropy(probs, normalize=True)

        # Should normalize to [0.25, 0.25, 0.25, 0.25]
        expected = np.log(4)
        assert np.isclose(entropy, expected, rtol=1e-5)

    def test_entropy_non_normalized(self):
        """Test entropy without normalization."""
        from tokenpulse.core.info_theory import InfoTheoryCalculator

        # Non-normalized distribution
        probs = np.array([0.2, 0.2, 0.2])  # Sum = 0.6
        entropy = InfoTheoryCalculator.entropy(probs, normalize=False)

        # Should compute -sum(p * log(p)) directly
        expected = -np.sum(probs * np.log(probs + 1e-10))
        assert np.isclose(entropy, expected, rtol=1e-5)


class TestKLDivergence:
    """Tests for KL divergence calculation."""

    def test_kl_identical_distributions(self):
        """KL divergence of identical distributions should be zero."""
        from tokenpulse.core.info_theory import InfoTheoryCalculator

        p = np.array([0.25, 0.25, 0.25, 0.25])
        q = np.array([0.25, 0.25, 0.25, 0.25])

        kl = InfoTheoryCalculator.kl_divergence(p, q)
        assert np.isclose(kl, 0.0, atol=1e-10)

    def test_kl_non_symmetric(self):
        """KL divergence should be asymmetric: D_KL(P||Q) != D_KL(Q||P)."""
        from tokenpulse.core.info_theory import InfoTheoryCalculator

        p = np.array([0.8, 0.2])
        q = np.array([0.5, 0.5])

        kl_pq = InfoTheoryCalculator.kl_divergence(p, q)
        kl_qp = InfoTheoryCalculator.kl_divergence(q, p)

        assert not np.isclose(kl_pq, kl_qp)

    def test_kl_deterministic_p(self):
        """Test KL when P is deterministic."""
        from tokenpulse.core.info_theory import InfoTheoryCalculator

        p = np.array([1.0, 0.0])
        q = np.array([0.5, 0.5])

        kl = InfoTheoryCalculator.kl_divergence(p, q)
        expected = np.log(1.0 / 0.5)  # log(2)
        assert np.isclose(kl, expected, rtol=1e-5)

    def test_kl_batch_computation(self):
        """Test batch computation of KL divergence."""
        from tokenpulse.core.info_theory import InfoTheoryCalculator

        p_batch = np.array([
            [0.5, 0.5],
            [0.8, 0.2],
            [1.0, 0.0],
        ])
        q_batch = np.array([
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
        ])

        kl_values = InfoTheoryCalculator.kl_divergence(p_batch, q_batch, dim=1)

        assert len(kl_values) == 3
        assert np.isclose(kl_values[0], 0.0, atol=1e-10)

    def test_kl_shape_mismatch(self):
        """Test that shape mismatch raises error."""
        from tokenpulse.core.info_theory import InfoTheoryCalculator

        p = np.array([0.5, 0.5])
        q = np.array([0.33, 0.33, 0.34])

        with pytest.raises(ValueError):
            InfoTheoryCalculator.kl_divergence(p, q)


class TestCrossEntropy:
    """Tests for cross-entropy calculation."""

    def test_cross_entropy_identical_distributions(self):
        """Cross-entropy of identical distributions equals entropy."""
        from tokenpulse.core.info_theory import InfoTheoryCalculator

        p = np.array([0.25, 0.25, 0.25, 0.25])
        q = np.array([0.25, 0.25, 0.25, 0.25])

        ce = InfoTheoryCalculator.cross_entropy(p, q)
        h = InfoTheoryCalculator.entropy(p)

        assert np.isclose(ce, h, rtol=1e-5)

    def test_cross_entropy_relationship(self):
        """H(P,Q) = H(P) + D_KL(P||Q)."""
        from tokenpulse.core.info_theory import InfoTheoryCalculator

        p = np.array([0.7, 0.3])
        q = np.array([0.5, 0.5])

        ce = InfoTheoryCalculator.cross_entropy(p, q)
        h = InfoTheoryCalculator.entropy(p)
        kl = InfoTheoryCalculator.kl_divergence(p, q)

        assert np.isclose(ce, h + kl, rtol=1e-5)


class TestTopPMass:
    """Tests for top-p mass calculation."""

    def test_top_p_concentrated(self):
        """Concentrated distribution should need few tokens."""
        from tokenpulse.core.info_theory import InfoTheoryCalculator

        probs = np.array([0.9, 0.05, 0.03, 0.02])
        count = InfoTheoryCalculator.top_p_mass(probs, p=0.9)

        # Top token has 0.9 which equals p=0.9, need only 1 token
        assert count == 1

    def test_top_p_uniform(self):
        """Uniform distribution should need many tokens."""
        from tokenpulse.core.info_theory import InfoTheoryCalculator

        probs = np.array([0.25, 0.25, 0.25, 0.25])
        count = InfoTheoryCalculator.top_p_mass(probs, p=0.9)

        # Need 4 tokens: 0.25*3 = 0.75 < 0.9, 0.25*4 = 1.0 >= 0.9
        assert count == 4

    def test_top_p_various_thresholds(self):
        """Test different probability thresholds."""
        from tokenpulse.core.info_theory import InfoTheoryCalculator

        probs = np.array([0.4, 0.3, 0.2, 0.1])

        # Sorted: [0.4, 0.3, 0.2, 0.1], cumsum: [0.4, 0.7, ~0.9, 1.0]
        # Note: cumsum[2] ≈ 0.9 (float precision makes it slightly less than 0.9)

        count_50 = InfoTheoryCalculator.top_p_mass(probs, p=0.5)
        # cumsum >= 0.5: first at index 1 (0.7), so need 2 tokens
        assert count_50 == 2

        count_70 = InfoTheoryCalculator.top_p_mass(probs, p=0.7)
        # cumsum >= 0.7: first at index 1 (0.7), so need 2 tokens
        assert count_70 == 2

        count_89 = InfoTheoryCalculator.top_p_mass(probs, p=0.89)
        # cumsum >= 0.89: first at index 2 (~0.9), so need 3 tokens
        assert count_89 == 3

        count_91 = InfoTheoryCalculator.top_p_mass(probs, p=0.91)
        # cumsum >= 0.91: first at index 3 (1.0), so need 4 tokens
        assert count_91 == 4


class TestPerplexity:
    """Tests for perplexity calculation."""

    def test_perplexity_uniform(self):
        """Perplexity of uniform distribution = vocab size."""
        from tokenpulse.core.info_theory import InfoTheoryCalculator

        probs = np.array([0.25, 0.25, 0.25, 0.25])
        ppl = InfoTheoryCalculator.perplexity(probs)

        # exp(log(4)) = 4
        assert np.isclose(ppl, 4.0, rtol=1e-5)

    def test_perplexity_deterministic(self):
        """Perplexity of deterministic distribution = 1."""
        from tokenpulse.core.info_theory import InfoTheoryCalculator

        probs = np.array([1.0, 0.0, 0.0, 0.0])
        ppl = InfoTheoryCalculator.perplexity(probs)

        assert np.isclose(ppl, 1.0, rtol=1e-5)


class TestNormalizedEntropy:
    """Tests for normalized entropy calculation."""

    def test_normalized_entropy_range(self):
        """Normalized entropy should be between 0 and 1."""
        from tokenpulse.core.info_theory import InfoTheoryCalculator

        probs = np.array([0.5, 0.3, 0.2])
        ne = InfoTheoryCalculator.normalized_entropy(probs)

        assert 0.0 <= ne <= 1.0

    def test_normalized_entropy_uniform(self):
        """Normalized entropy of uniform distribution = 1."""
        from tokenpulse.core.info_theory import InfoTheoryCalculator

        probs = np.array([0.25, 0.25, 0.25, 0.25])
        ne = InfoTheoryCalculator.normalized_entropy(probs)

        assert np.isclose(ne, 1.0, rtol=1e-5)

    def test_normalized_entropy_deterministic(self):
        """Normalized entropy of deterministic distribution should be near 0."""
        from tokenpulse.core.info_theory import InfoTheoryCalculator

        probs = np.array([1.0, 0.0, 0.0, 0.0])
        ne = InfoTheoryCalculator.normalized_entropy(probs)

        # Should be very close to zero (within numerical precision)
        assert ne < 1e-8


class TestEntropyWithBase:
    """Tests for entropy calculation with different logarithm bases."""

    def test_entropy_base_2(self):
        """Test entropy with base 2 (bits)."""
        from tokenpulse.core.info_theory import InfoTheoryCalculator

        # Fair coin flip: 1 bit of entropy
        probs = np.array([0.5, 0.5])
        entropy_bits = InfoTheoryCalculator.entropy(probs, base=2)
        assert np.isclose(entropy_bits, 1.0, rtol=1e-5)

    def test_entropy_base_2_uniform(self):
        """Test entropy of uniform distribution with base 2."""
        from tokenpulse.core.info_theory import InfoTheoryCalculator

        # 4 equally likely outcomes: 2 bits of entropy
        probs = np.array([0.25, 0.25, 0.25, 0.25])
        entropy_bits = InfoTheoryCalculator.entropy(probs, base=2)
        assert np.isclose(entropy_bits, 2.0, rtol=1e-5)

    def test_entropy_base_conversion(self):
        """Test that base conversion is correct."""
        from tokenpulse.core.info_theory import InfoTheoryCalculator

        probs = np.array([0.5, 0.5])

        entropy_natural = InfoTheoryCalculator.entropy(probs)  # ln
        entropy_base2 = InfoTheoryCalculator.entropy(probs, base=2)

        # H_base2 = H_natural / ln(2)
        expected = entropy_natural / np.log(2)
        assert np.isclose(entropy_base2, expected, rtol=1e-5)
