"""
L1 Layer Data Structures

Defines the unified output format for model providers.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import torch


@dataclass
class ModelCapabilities:
    """
    Model capability declaration.

    Describes what features a model provider supports.
    """

    supports_logits: bool = True
    supports_hidden_states: bool = True
    supports_attention_weights: bool = True
    max_batch_size: int = 1
    supports_streaming: bool = False

    def check_capability(self, capability: str) -> bool:
        """
        Check if a specific capability is supported.

        Args:
            capability: Name of the capability to check (e.g., 'supports_logits')

        Returns:
            True if capability is supported, False otherwise
        """
        return getattr(self, capability, False)


@dataclass
class ModelOutput:
    """
    Unified model output structure.

    This is the standard output format that all providers must produce,
    ensuring consistency across different model backends.
    """

    # Basic information
    model_id: str
    prompt: str
    generated_text: str

    # Probability distribution
    logits: Optional[torch.Tensor] = None  # [seq_len, vocab_size]
    logprobs: Optional[torch.Tensor] = None  # [seq_len, vocab_size]
    top_logprobs: Optional[List[Dict[int, float]]] = None

    # Intermediate activations
    hidden_states: Optional[Dict[int, torch.Tensor]] = field(
        default_factory=dict
    )  # layer_id -> [seq_len, hidden_dim]
    attention_weights: Optional[Dict[int, torch.Tensor]] = field(
        default_factory=dict
    )  # layer_id -> [heads, seq_len, seq_len]

    # Metadata
    tokens: Optional[List[int]] = field(default_factory=list)
    token_strings: Optional[List[str]] = field(default_factory=list)

    # Additional info
    metadata: Dict[str, Any] = field(default_factory=dict)

    def has_logits(self) -> bool:
        """Check if logits are available."""
        return self.logits is not None

    def has_hidden_states(self) -> bool:
        """Check if hidden states are available."""
        return self.hidden_states is not None and len(self.hidden_states) > 0

    def has_attention_weights(self) -> bool:
        """Check if attention weights are available."""
        return self.attention_weights is not None and len(self.attention_weights) > 0

    def get_layer_hidden_states(self, layer_idx: int) -> Optional[torch.Tensor]:
        """
        Get hidden states for a specific layer.

        Args:
            layer_idx: Layer index

        Returns:
            Hidden states tensor or None if not available
        """
        if self.hidden_states is None:
            return None
        return self.hidden_states.get(layer_idx)

    def get_layer_attention(self, layer_idx: int) -> Optional[torch.Tensor]:
        """
        Get attention weights for a specific layer.

        Args:
            layer_idx: Layer index

        Returns:
            Attention weights tensor or None if not available
        """
        if self.attention_weights is None:
            return None
        return self.attention_weights.get(layer_idx)

    def to_cpu(self) -> "ModelOutput":
        """
        Move all tensors to CPU.

        Returns:
            New ModelOutput with CPU tensors
        """
        def move_tensor(t):
            if t is None:
                return None
            if isinstance(t, torch.Tensor):
                return t.detach().cpu()
            return t

        return ModelOutput(
            model_id=self.model_id,
            prompt=self.prompt,
            generated_text=self.generated_text,
            logits=move_tensor(self.logits),
            logprobs=move_tensor(self.logprobs),
            top_logprobs=self.top_logprobs,
            hidden_states={k: move_tensor(v) for k, v in (self.hidden_states or {}).items()},
            attention_weights={k: move_tensor(v) for k, v in (self.attention_weights or {}).items()},
            tokens=self.tokens,
            token_strings=self.token_strings,
            metadata=self.metadata,
        )
