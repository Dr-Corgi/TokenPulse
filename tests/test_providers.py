"""
Unit tests for Model Providers (L1 Layer).

Tests cover:
- ModelOutput data structure
- ModelCapabilities data structure
- BaseProvider interface
"""

import pytest
import numpy as np


class TestModelOutput:
    """Tests for ModelOutput data structure."""

    def test_model_output_creation(self):
        """Test basic ModelOutput creation."""
        from tokenpulse.providers.data_structures import ModelOutput

        output = ModelOutput(
            model_id="test-model",
            prompt="Hello",
            generated_text="Hello world"
        )

        assert output.model_id == "test-model"
        assert output.prompt == "Hello"
        assert output.generated_text == "Hello world"
        assert output.logits is None
        assert output.hidden_states is not None
        assert len(output.hidden_states) == 0

    def test_model_output_with_logits(self):
        """Test ModelOutput with logits."""
        try:
            import torch
            from tokenpulse.providers.data_structures import ModelOutput

            logits = torch.randn(10, 1000)  # [seq_len, vocab_size]
            output = ModelOutput(
                model_id="test-model",
                prompt="Test",
                generated_text="Test output",
                logits=logits
            )

            assert output.has_logits()
            assert output.logits.shape == (10, 1000)

        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_model_output_with_hidden_states(self):
        """Test ModelOutput with hidden states."""
        try:
            import torch
            from tokenpulse.providers.data_structures import ModelOutput

            hidden_states = {
                0: torch.randn(5, 256),
                1: torch.randn(5, 256),
                2: torch.randn(5, 256),
            }

            output = ModelOutput(
                model_id="test-model",
                prompt="Test",
                generated_text="Test",
                hidden_states=hidden_states
            )

            assert output.has_hidden_states()
            assert len(output.hidden_states) == 3
            assert output.get_layer_hidden_states(1) is not None
            assert output.get_layer_hidden_states(10) is None

        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_model_output_to_cpu(self):
        """Test moving ModelOutput to CPU."""
        try:
            import torch
            from tokenpulse.providers.data_structures import ModelOutput

            logits = torch.randn(5, 100)
            hidden_states = {0: torch.randn(5, 64)}

            output = ModelOutput(
                model_id="test-model",
                prompt="Test",
                generated_text="Test",
                logits=logits,
                hidden_states=hidden_states
            )

            cpu_output = output.to_cpu()

            assert cpu_output.logits.device.type == "cpu"
            assert cpu_output.hidden_states[0].device.type == "cpu"

        except ImportError:
            pytest.skip("PyTorch not installed")


class TestModelCapabilities:
    """Tests for ModelCapabilities data structure."""

    def test_default_capabilities(self):
        """Test default capability settings."""
        from tokenpulse.providers.data_structures import ModelCapabilities

        caps = ModelCapabilities()

        assert caps.supports_logits is True
        assert caps.supports_hidden_states is True
        assert caps.supports_attention_weights is True
        assert caps.max_batch_size == 1
        assert caps.supports_streaming is False

    def test_check_capability(self):
        """Test capability checking."""
        from tokenpulse.providers.data_structures import ModelCapabilities

        caps = ModelCapabilities(
            supports_logits=True,
            supports_hidden_states=False
        )

        assert caps.check_capability("supports_logits") is True
        assert caps.check_capability("supports_hidden_states") is False
        assert caps.check_capability("nonexistent_capability") is False


class TestBaseProvider:
    """Tests for BaseProvider abstract class."""

    def test_cannot_instantiate(self):
        """BaseProvider should not be directly instantiable."""
        from tokenpulse.providers.base import BaseProvider

        with pytest.raises(TypeError):
            BaseProvider("test-model")

    def test_validate_capabilities(self):
        """Test capability validation in subclasses."""
        from tokenpulse.providers.base import BaseProvider
        from tokenpulse.providers.data_structures import ModelCapabilities

        class MockProvider(BaseProvider):
            def __init__(self, model_id):
                super().__init__(model_id)
                self._capabilities = ModelCapabilities(
                    supports_logits=True,
                    supports_hidden_states=False
                )

            @property
            def capabilities(self):
                return self._capabilities

            def generate(self, prompts, **kwargs):
                pass

            def tokenize(self, text):
                return []

            def decode(self, tokens):
                return ""

        provider = MockProvider("test")

        # Should pass
        provider.validate_capabilities(need_logits=True)

        # Should fail
        with pytest.raises(ValueError):
            provider.validate_capabilities(need_hidden_states=True)
