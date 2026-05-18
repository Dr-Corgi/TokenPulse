"""
Information Theory Calculator - L3 Layer Core Engine.

Provides entropy, KL divergence, and related information-theoretic computations.
These functions form the foundation for probability distribution analysis.
"""

from typing import Union, Tuple, Optional
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class InfoTheoryCalculator:
    """
    Information theory computation utilities.

    All methods support both NumPy arrays and PyTorch tensors.
    These are pure functions with no side effects, making them easy to test
    and reuse across different diagnostic checks.

    Example:
        >>> import numpy as np
        >>> probs = np.array([0.5, 0.3, 0.2])
        >>> entropy = InfoTheoryCalculator.entropy(probs)
        >>> print(f"Entropy: {entropy:.4f}")
    """

    @staticmethod
    def entropy(
        prob_dist: Union[np.ndarray, "torch.Tensor"],
        dim: int = -1,
        normalize: bool = True,
        base: Optional[float] = None
    ) -> Union[float, np.ndarray]:
        """
        Calculate Shannon entropy: H(X) = -Σ p(x) * log(p(x))

        Entropy measures the uncertainty or randomness in a probability distribution.
        Higher entropy = more uncertain/uniform distribution.
        Lower entropy = more concentrated/predictable distribution.

        Args:
            prob_dist: Probability distribution tensor/array.
                      Shape [..., vocab_size] for batch computation.
            dim: Dimension along which to compute entropy (default: -1, last dimension)
            normalize: If True, normalize inputs to sum to 1
            base: Logarithm base. If None, use natural log. Use 2 for bits, 10 for dits.

        Returns:
            Entropy value(s). Scalar if input is 1D, otherwise array of entropies.

        Examples:
            >>> probs = np.array([0.25, 0.25, 0.25, 0.25])
            >>> InfoTheoryCalculator.entropy(probs)
            1.386294...  # log(4) ≈ 1.386

            >>> probs = np.array([0.5, 0.5])
            >>> InfoTheoryCalculator.entropy(probs, base=2)
            1.0  # 1 bit of entropy

            >>> probs = np.array([1.0, 0.0, 0.0])  # Deterministic
            >>> InfoTheoryCalculator.entropy(probs)
            0.0  # No uncertainty
        """
        if HAS_TORCH and isinstance(prob_dist, torch.Tensor):
            return InfoTheoryCalculator._entropy_torch(prob_dist, dim, normalize, base)
        else:
            return InfoTheoryCalculator._entropy_numpy(prob_dist, dim, normalize, base)

    @staticmethod
    def _entropy_numpy(
        prob_dist: np.ndarray,
        dim: int = -1,
        normalize: bool = True,
        base: Optional[float] = None
    ) -> Union[float, np.ndarray]:
        """NumPy implementation of entropy calculation."""
        prob_dist = np.asarray(prob_dist, dtype=np.float64)

        # Normalize if needed
        if normalize:
            sum_val = np.sum(prob_dist, axis=dim, keepdims=True)
            sum_val = np.where(sum_val == 0, 1, sum_val)  # Avoid division by zero
            prob_dist = prob_dist / sum_val

        # Clamp to avoid log(0)
        prob_dist = np.clip(prob_dist, 1e-10, 1.0)

        # Compute entropy: -Σ p * log(p)
        log_probs = np.log(prob_dist)
        entropy = -np.sum(prob_dist * log_probs, axis=dim)

        # Convert to desired base if specified
        if base is not None:
            entropy = entropy / np.log(base)

        return np.abs(entropy)

    @staticmethod
    def _entropy_torch(
        prob_dist: "torch.Tensor",
        dim: int = -1,
        normalize: bool = True,
        base: Optional[float] = None
    ) -> Union[float, "torch.Tensor"]:
        """PyTorch implementation of entropy calculation."""
        prob_dist = prob_dist.float()

        # Normalize if needed
        if normalize:
            sum_val = prob_dist.sum(dim=dim, keepdim=True)
            sum_val = sum_val.clamp(min=1e-10)  # Avoid division by zero
            prob_dist = prob_dist / sum_val

        # Clamp to avoid log(0)
        prob_dist = prob_dist.clamp(min=1e-10, max=1.0)

        # Compute entropy
        log_probs = torch.log(prob_dist)
        entropy = -torch.sum(prob_dist * log_probs, dim=dim)

        # Convert to desired base if specified
        if base is not None:
            entropy = entropy / np.log(base)

        return torch.abs(entropy)

    @staticmethod
    def kl_divergence(
        p: Union[np.ndarray, "torch.Tensor"],
        q: Union[np.ndarray, "torch.Tensor"],
        dim: int = -1,
        normalize: bool = True
    ) -> Union[float, np.ndarray]:
        """
        Calculate Kullback-Leibler divergence: D_KL(P || Q) = Σ P(x) * log(P(x) / Q(x))

        KL divergence measures how one probability distribution diverges from another.
        It is asymmetric: D_KL(P || Q) ≠ D_KL(Q || P).
        D_KL = 0 when P and Q are identical.

        Args:
            p: First probability distribution P
            q: Second probability distribution Q (reference)
            dim: Dimension for computation
            normalize: If True, normalize inputs to sum to 1

        Returns:
            KL divergence value(s). Non-negative.

        Raises:
            ValueError: If shapes don't match

        Examples:
            >>> p = np.array([0.5, 0.5])
            >>> q = np.array([0.5, 0.5])
            >>> InfoTheoryCalculator.kl_divergence(p, q)
            0.0  # Identical distributions

            >>> p = np.array([1.0, 0.0])
            >>> q = np.array([0.5, 0.5])
            >>> InfoTheoryCalculator.kl_divergence(p, q)
            0.693...  # log(2)
        """
        if HAS_TORCH and isinstance(p, torch.Tensor) and isinstance(q, torch.Tensor):
            return InfoTheoryCalculator._kl_divergence_torch(p, q, dim, normalize)
        else:
            p_np = p.detach().cpu().numpy() if HAS_TORCH and isinstance(p, torch.Tensor) else p
            q_np = q.detach().cpu().numpy() if HAS_TORCH and isinstance(q, torch.Tensor) else q
            return InfoTheoryCalculator._kl_divergence_numpy(p_np, q_np, dim, normalize)

    @staticmethod
    def _kl_divergence_numpy(
        p: np.ndarray,
        q: np.ndarray,
        dim: int = -1,
        normalize: bool = True
    ) -> Union[float, np.ndarray]:
        """NumPy implementation of KL divergence."""
        p = np.asarray(p, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64)

        if p.shape != q.shape:
            raise ValueError(f"Shape mismatch: p {p.shape} vs q {q.shape}")

        # Normalize if needed
        if normalize:
            p_sum = np.sum(p, axis=dim, keepdims=True)
            p_sum = np.where(p_sum == 0, 1, p_sum)
            p = p / p_sum

            q_sum = np.sum(q, axis=dim, keepdims=True)
            q_sum = np.where(q_sum == 0, 1, q_sum)
            q = q / q_sum

        # Clamp to avoid log(0) and division by zero
        p = np.clip(p, 1e-10, 1.0)
        q = np.clip(q, 1e-10, 1.0)

        # KL divergence: Σ p * log(p/q)
        log_ratio = np.log(p) - np.log(q)
        kl = np.sum(p * log_ratio, axis=dim)

        return kl

    @staticmethod
    def _kl_divergence_torch(
        p: "torch.Tensor",
        q: "torch.Tensor",
        dim: int = -1,
        normalize: bool = True
    ) -> Union[float, "torch.Tensor"]:
        """PyTorch implementation of KL divergence."""
        p = p.float()
        q = q.float()

        if p.shape != q.shape:
            raise ValueError(f"Shape mismatch: p {p.shape} vs q {q.shape}")

        # Normalize if needed
        if normalize:
            p = p / p.sum(dim=dim, keepdim=True).clamp(min=1e-10)
            q = q / q.sum(dim=dim, keepdim=True).clamp(min=1e-10)

        # Clamp
        p = p.clamp(min=1e-10, max=1.0)
        q = q.clamp(min=1e-10, max=1.0)

        # KL divergence
        log_ratio = torch.log(p) - torch.log(q)
        kl = torch.sum(p * log_ratio, dim=dim)

        return kl

    @staticmethod
    def cross_entropy(
        p: Union[np.ndarray, "torch.Tensor"],
        q: Union[np.ndarray, "torch.Tensor"],
        dim: int = -1,
        normalize: bool = True
    ) -> Union[float, np.ndarray]:
        """
        Calculate cross-entropy: H(P, Q) = -Σ P(x) * log(Q(x))

        Cross-entropy measures the average number of bits needed to identify
        an event from distribution P when using a code optimized for Q.

        Relationship: H(P, Q) = H(P) + D_KL(P || Q)

        Args:
            p: True probability distribution P
            q: Predicted probability distribution Q
            dim: Dimension for computation
            normalize: If True, normalize inputs

        Returns:
            Cross-entropy value(s)
        """
        if HAS_TORCH and isinstance(p, torch.Tensor) and isinstance(q, torch.Tensor):
            p = p.float()
            q = q.float()

            if normalize:
                p = p / p.sum(dim=dim, keepdim=True).clamp(min=1e-10)
                q = q / q.sum(dim=dim, keepdim=True).clamp(min=1e-10)

            q = q.clamp(min=1e-10)
            ce = -torch.sum(p * torch.log(q), dim=dim)
            return ce
        else:
            p_np = p.detach().cpu().numpy() if HAS_TORCH and isinstance(p, torch.Tensor) else p
            q_np = q.detach().cpu().numpy() if HAS_TORCH and isinstance(q, torch.Tensor) else q
            p_np = np.asarray(p_np, dtype=np.float64)
            q_np = np.asarray(q_np, dtype=np.float64)

            if normalize:
                p_np = p_np / np.sum(p_np, axis=dim, keepdims=True).clip(1e-10)
                q_np = q_np / np.sum(q_np, axis=dim, keepdims=True).clip(1e-10)

            q_np = np.clip(q_np, 1e-10, 1.0)
            ce = -np.sum(p_np * np.log(q_np), axis=dim)
            return ce

    @staticmethod
    def top_p_mass(
        prob_dist: Union[np.ndarray, "torch.Tensor"],
        p: float = 0.9,
        dim: int = -1
    ) -> Union[int, np.ndarray]:
        """
        Calculate the number of tokens needed to reach a cumulative probability p.

        This is useful for measuring how concentrated the probability mass is.
        Lower count = more concentrated (model is confident)
        Higher count = more spread out (model is uncertain)

        Args:
            prob_dist: Probability distribution
            p: Target cumulative probability (default: 0.9)
            dim: Dimension for computation

        Returns:
            Number of tokens needed to reach or exceed probability p

        Examples:
            >>> probs = np.array([0.8, 0.1, 0.05, 0.05])
            >>> InfoTheoryCalculator.top_p_mass(probs, p=0.9)
            2  # Top 2 tokens cover 90% probability (0.8 + 0.1 = 0.9)
        """
        if HAS_TORCH and isinstance(prob_dist, torch.Tensor):
            prob_dist = prob_dist.float()
            sorted_probs, _ = torch.sort(prob_dist, dim=dim, descending=True)
            cumsum = torch.cumsum(sorted_probs, dim=dim)

            # Find first index where cumsum >= p
            # Return k+1 where k is that index (1-indexed count)
            if cumsum.dim() == 1:
                indices = torch.where(cumsum >= p)[0]
                if len(indices) > 0:
                    return indices[0].item() + 1
                return len(cumsum)
            else:
                # Batch version
                result = []
                for i in range(cumsum.shape[0]):
                    indices = torch.where(cumsum[i] >= p)[0]
                    if len(indices) > 0:
                        result.append(indices[0].item() + 1)
                    else:
                        result.append(cumsum.shape[1])
                return torch.tensor(result)
        else:
            prob_dist = np.asarray(prob_dist, dtype=np.float64)
            sorted_probs = np.sort(prob_dist, axis=dim)[::-1]
            cumsum = np.cumsum(sorted_probs, axis=dim)

            # Find first index where cumsum >= p
            def find_first_ge(arr, threshold):
                indices = np.where(arr >= threshold)[0]
                return indices[0] + 1 if len(indices) > 0 else len(arr)

            if cumsum.ndim == 1:
                return find_first_ge(cumsum, p)
            else:
                return np.array([find_first_ge(row, p) for row in cumsum])

    @staticmethod
    def perplexity(
        prob_dist: Union[np.ndarray, "torch.Tensor"],
        dim: int = -1
    ) -> Union[float, np.ndarray]:
        """
        Calculate perplexity: PP(P) = exp(H(P))

        Perplexity is the exponentiated entropy, often used to evaluate
        language models. Lower perplexity = better model (more confident predictions).

        Args:
            prob_dist: Probability distribution
            dim: Dimension for computation

        Returns:
            Perplexity value(s)
        """
        entropy = InfoTheoryCalculator.entropy(prob_dist, dim=dim)
        return np.exp(entropy)

    @staticmethod
    def normalized_entropy(
        prob_dist: Union[np.ndarray, "torch.Tensor"],
        dim: int = -1
    ) -> Union[float, np.ndarray]:
        """
        Calculate normalized entropy: H_norm = H / H_max = H / log(n)

        Normalized entropy is between 0 and 1, where:
        - 0 means deterministic (all mass on one outcome)
        - 1 means uniform distribution

        Args:
            prob_dist: Probability distribution
            dim: Dimension for computation

        Returns:
            Normalized entropy value(s) between 0 and 1
        """
        entropy = InfoTheoryCalculator.entropy(prob_dist, dim=dim)

        # Get vocabulary size
        if HAS_TORCH and isinstance(prob_dist, torch.Tensor):
            vocab_size = prob_dist.shape[dim]
            max_entropy = np.log(vocab_size)

            if isinstance(entropy, torch.Tensor):
                return entropy / max_entropy
            else:
                return float(entropy) / max_entropy
        else:
            vocab_size = prob_dist.shape[dim]
            max_entropy = np.log(vocab_size)
            return entropy / max_entropy
