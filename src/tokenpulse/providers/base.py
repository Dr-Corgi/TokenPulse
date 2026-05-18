"""
Base Provider - Abstract base class for model providers.

All model adapters (TransformerLens, HuggingFace, vLLM, API) must inherit from this class.
"""

from abc import ABC, abstractmethod
from typing import List, Union, Optional

from tokenpulse.providers.data_structures import ModelOutput, ModelCapabilities


class BaseProvider(ABC):
    """
    Abstract base class for model providers.

    This class defines the interface that all model adapters must implement.
    The unified interface allows diagnostic checks to work with any backend.
    """

    def __init__(self, model_id: str, **kwargs):
        """
        Initialize the provider.

        Args:
            model_id: Model identifier (e.g., "gpt2-small", "meta-llama/Llama-3-8b")
            **kwargs: Additional provider-specific configuration
        """
        self.model_id = model_id
        self.config = kwargs
        self._capabilities: Optional[ModelCapabilities] = None

    @property
    @abstractmethod
    def capabilities(self) -> ModelCapabilities:
        """
        Return model capability declaration.

        Returns:
            ModelCapabilities instance describing what features are supported
        """
        pass

    @abstractmethod
    def generate(
        self,
        prompts: Union[str, List[str]],
        return_logits: bool = True,
        return_hidden_states: bool = False,
        return_attention_weights: bool = False,
        hidden_state_layers: Optional[List[int]] = None,
        **kwargs
    ) -> Union[ModelOutput, List[ModelOutput]]:
        """
        Generate text and extract features.

        This is the core method that all providers must implement.
        It generates text from prompts and optionally extracts intermediate activations.

        Args:
            prompts: Single prompt or list of prompts
            return_logits: Whether to return logits (default: True)
            return_hidden_states: Whether to return hidden states (default: False)
            return_attention_weights: Whether to return attention weights (default: False)
            hidden_state_layers: Specific layers to extract hidden states from.
                                None means all layers.
            **kwargs: Additional generation parameters (temperature, max_tokens, etc.)

        Returns:
            ModelOutput for single prompt, or List[ModelOutput] for multiple prompts

        Raises:
            ValueError: If requested features are not supported by this provider
        """
        pass

    @abstractmethod
    def tokenize(self, text: str) -> List[int]:
        """
        Tokenize text into token IDs.

        Args:
            text: Text to tokenize

        Returns:
            List of token IDs
        """
        pass

    @abstractmethod
    def decode(self, tokens: List[int]) -> str:
        """
        Decode token IDs back to text.

        Args:
            tokens: List of token IDs

        Returns:
            Decoded text string
        """
        pass

    def check_capability(self, requirement: str) -> bool:
        """
        Check if the provider supports a specific capability.

        Args:
            requirement: Capability name (e.g., 'supports_logits')

        Returns:
            True if capability is supported
        """
        return self.capabilities.check_capability(requirement)

    def validate_capabilities(
        self,
        need_logits: bool = False,
        need_hidden_states: bool = False,
        need_attention_weights: bool = False,
    ) -> None:
        """
        Validate that the provider supports requested features.

        Args:
            need_logits: Whether logits are needed
            need_hidden_states: Whether hidden states are needed
            need_attention_weights: Whether attention weights are needed

        Raises:
            ValueError: If a required capability is not supported
        """
        caps = self.capabilities

        if need_logits and not caps.supports_logits:
            raise ValueError(
                f"Provider '{self.model_id}' does not support logits extraction"
            )
        if need_hidden_states and not caps.supports_hidden_states:
            raise ValueError(
                f"Provider '{self.model_id}' does not support hidden states extraction"
            )
        if need_attention_weights and not caps.supports_attention_weights:
            raise ValueError(
                f"Provider '{self.model_id}' does not support attention weights extraction"
            )

    def get_vocab_size(self) -> int:
        """
        Get the vocabulary size of the model.

        Returns:
            Vocabulary size

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses should implement get_vocab_size()")

    def get_num_layers(self) -> int:
        """
        Get the number of layers in the model.

        Returns:
            Number of layers

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses should implement get_num_layers()")

    def get_hidden_dim(self) -> int:
        """
        Get the hidden dimension of the model.

        Returns:
            Hidden dimension size

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses should implement get_hidden_dim()")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_id='{self.model_id}')"
