"""
TransformerLens Provider - Adapter for TransformerLens models.

TransformerLens provides easy access to intermediate activations (attention patterns,
hidden states, etc.) through its HookedTransformer architecture.
"""

from typing import List, Union, Optional, Dict, Any
import torch

from tokenpulse.providers.base import BaseProvider
from tokenpulse.providers.data_structures import ModelOutput, ModelCapabilities


class TransformerLensProvider(BaseProvider):
    """
    TransformerLens model provider.

    This provider uses TransformerLens's HookedTransformer for easy extraction
    of logits, hidden states, and attention weights.

    Example:
        >>> provider = TransformerLensProvider("gpt2-small")
        >>> output = provider.generate("Hello, world!", return_hidden_states=True)
        >>> print(output.logits.shape)
        >>> print(output.hidden_states.keys())
    """

    def __init__(
        self,
        model_id: str,
        device: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize TransformerLens provider.

        Args:
            model_id: Model name (e.g., "gpt2-small", "pythia-70m")
            device: Device to run on ("cuda", "cpu", or None for auto)
            **kwargs: Additional arguments passed to HookedTransformer.from_pretrained()
        """
        super().__init__(model_id, **kwargs)

        # Lazy import to avoid dependency issues
        try:
            from transformer_lens import HookedTransformer
        except ImportError as e:
            raise ImportError(
                "TransformerLens is required for TransformerLensProvider. "
                "Install it with: pip install transformer-lens"
            ) from e

        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Load model - from_pretrained handles device placement via device parameter
        self.model = HookedTransformer.from_pretrained(
            model_id,
            device=device,
            **kwargs
        )
        self.model.eval()

        # Get model info from config
        # Note: n_heads defaults to -1 in config and is auto-set to d_model // d_head
        self._vocab_size = self.model.cfg.d_vocab
        self._n_layers = self.model.cfg.n_layers
        self._hidden_dim = self.model.cfg.d_model

        # Handle n_heads which may be -1 (auto-computed)
        n_heads = self.model.cfg.n_heads
        if n_heads == -1:
            # Auto-computed as d_model // d_head
            n_heads = self.model.cfg.d_model // self.model.cfg.d_head
        self._n_heads = n_heads

        # Handle d_vocab which may be -1 for some models
        if self._vocab_size == -1:
            self._vocab_size = self.model.cfg.d_vocab_out

        # Set capabilities
        self._capabilities = ModelCapabilities(
            supports_logits=True,
            supports_hidden_states=True,
            supports_attention_weights=True,
            max_batch_size=kwargs.get("max_batch_size", 1),
            supports_streaming=False,
        )

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    def generate(
        self,
        prompts: Union[str, List[str]],
        return_logits: bool = True,
        return_hidden_states: bool = False,
        return_attention_weights: bool = False,
        hidden_state_layers: Optional[List[int]] = None,
        max_new_tokens: int = 10,
        **kwargs
    ) -> Union[ModelOutput, List[ModelOutput]]:
        """
        Generate text and extract features using run_with_cache.

        Args:
            prompts: Single prompt or list of prompts
            return_logits: Whether to return logits
            return_hidden_states: Whether to return hidden states
            return_attention_weights: Whether to return attention weights
            hidden_state_layers: Specific layers for hidden states (None = all)
            max_new_tokens: Maximum new tokens to generate
            **kwargs: Additional generation parameters

        Returns:
            ModelOutput or List[ModelOutput]
        """
        # Validate capabilities
        self.validate_capabilities(
            need_logits=return_logits,
            need_hidden_states=return_hidden_states,
            need_attention_weights=return_attention_weights,
        )

        # Handle single vs batch
        single_input = isinstance(prompts, str)
        if single_input:
            prompts = [prompts]

        results = []
        for prompt in prompts:
            result = self._generate_single(
                prompt=prompt,
                return_logits=return_logits,
                return_hidden_states=return_hidden_states,
                return_attention_weights=return_attention_weights,
                hidden_state_layers=hidden_state_layers,
                max_new_tokens=max_new_tokens,
                **kwargs
            )
            results.append(result)

        return results[0] if single_input else results

    def _generate_single(
        self,
        prompt: str,
        return_logits: bool,
        return_hidden_states: bool,
        return_attention_weights: bool,
        hidden_state_layers: Optional[List[int]],
        max_new_tokens: int,
        **kwargs
    ) -> ModelOutput:
        """Generate for a single prompt."""

        # Tokenize input to get input length
        input_tokens = self.model.to_tokens(prompt)
        input_len = input_tokens.shape[1]

        # Run model with cache to get all activations
        # run_with_cache returns (logits, ActivationCache)
        with torch.no_grad():
            logits, cache = self.model.run_with_cache(
                prompt,
                return_type="logits"
            )

        # Generate text - return tokens to get generated token IDs
        with torch.no_grad():
            generated_tokens = self.model.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                return_type="tokens",  # Important: return tokens, not string
                **kwargs
            )

        # Decode generated tokens to text
        generated_text = self.model.to_string(generated_tokens)

        # Extract hidden states using shorthand tuple keys (more reliable than string keys)
        # ActivationCache supports: cache[("resid_post", layer_idx)]
        hidden_states: Dict[int, torch.Tensor] = {}
        if return_hidden_states:
            all_layers = hidden_state_layers or list(range(self._n_layers))
            for layer_idx in all_layers:
                try:
                    # Use shorthand tuple access - handles negative indexing too
                    resid = cache[("resid_post", layer_idx)]
                    hidden_states[layer_idx] = resid.squeeze(0)
                except KeyError:
                    # Layer might not exist for this model
                    pass

        # Extract attention weights using shorthand tuple keys
        # cache[("pattern", layer_idx)] for attention patterns
        attention_weights: Dict[int, torch.Tensor] = {}
        if return_attention_weights:
            for layer_idx in range(self._n_layers):
                try:
                    # Use shorthand tuple access
                    pattern = cache[("pattern", layer_idx)]
                    attention_weights[layer_idx] = pattern.squeeze(0)
                except KeyError:
                    # Layer might not exist or model is attn_only
                    pass

        # Build output
        output = ModelOutput(
            model_id=self.model_id,
            prompt=prompt,
            generated_text=generated_text,
            logits=logits.squeeze(0) if return_logits else None,
            logprobs=torch.log_softmax(logits.squeeze(0), dim=-1) if return_logits else None,
            hidden_states=hidden_states,
            attention_weights=attention_weights,
            tokens=generated_tokens.squeeze(0).tolist() if generated_tokens.dim() > 1 else generated_tokens.tolist(),
            token_strings=self.model.to_str_tokens(generated_tokens.squeeze(0)),
            metadata={
                "input_length": input_len,
                "total_length": generated_tokens.shape[1] if generated_tokens.dim() > 1 else len(generated_tokens),
            }
        )

        return output

    def tokenize(self, text: str) -> List[int]:
        """Tokenize text using the model's tokenizer."""
        tokens = self.model.to_tokens(text)
        return tokens.squeeze(0).tolist()

    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs back to text."""
        # to_string accepts list or tensor, handles device internally
        return self.model.to_string(tokens)

    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return self._vocab_size

    def get_num_layers(self) -> int:
        """Get number of layers."""
        return self._n_layers

    def get_hidden_dim(self) -> int:
        """Get hidden dimension."""
        return self._hidden_dim

    def get_attention_heads(self) -> int:
        """Get number of attention heads."""
        return self._n_heads
