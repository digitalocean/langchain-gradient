"""Unit tests for ChatGradient with rate limiting (no API calls needed)."""

from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import HumanMessage

from langchain_gradient.chat_models import ChatGradient


class TestChatGradientRateLimiting:
    """Test rate limiting integration in ChatGradient."""

    def test_rate_limiting_disabled_by_default(self):
        """Test that rate limiting is disabled by default."""
        llm = ChatGradient(api_key="fake-key")

        assert llm.enable_rate_limiting is False
        assert llm._rate_limiter is None

    def test_rate_limiting_can_be_enabled(self):
        """Test that rate limiting can be enabled."""
        llm = ChatGradient(
            api_key="fake-key", enable_rate_limiting=True, max_requests_per_minute=60
        )

        assert llm.enable_rate_limiting is True
        assert llm._rate_limiter is not None
        assert llm._rate_limiter.max_requests == 60
        assert llm._rate_limiter.time_window == 60.0

    def test_custom_rate_limit(self):
        """Test that custom rate limits are applied."""
        llm = ChatGradient(
            api_key="fake-key", enable_rate_limiting=True, max_requests_per_minute=30
        )

        assert llm._rate_limiter.max_requests == 30

    def test_invalid_rate_limit_raises_error(self):
        """Test that invalid rate limits raise ValueError."""
        with pytest.raises(
            ValueError, match="max_requests_per_minute must be positive"
        ):
            ChatGradient(
                api_key="fake-key", enable_rate_limiting=True, max_requests_per_minute=0
            )

        with pytest.raises(
            ValueError, match="max_requests_per_minute must be positive"
        ):
            ChatGradient(
                api_key="fake-key",
                enable_rate_limiting=True,
                max_requests_per_minute=-10,
            )

    @patch("langchain_gradient.chat_models.Gradient")
    def test_rate_limiter_called_during_generate(self, mock_gradient):
        """Test that rate limiter is called during _generate."""
        # Setup mock
        mock_client = Mock()
        mock_gradient.return_value = mock_client

        mock_completion = Mock()
        mock_completion.choices = [Mock()]
        mock_completion.choices[0].message.content = "Test response"
        mock_completion.choices[0].finish_reason = "stop"
        mock_completion.usage = Mock(
            prompt_tokens=10, completion_tokens=20, total_tokens=30
        )
        mock_completion.model = "llama3.3-70b-instruct"
        mock_completion.id = "test-id"

        mock_client.chat.completions.create.return_value = mock_completion

        # Create LLM with rate limiting
        llm = ChatGradient(
            api_key="fake-key", enable_rate_limiting=True, max_requests_per_minute=5
        )

        # Spy on rate limiter
        original_wait = llm._rate_limiter.wait_if_needed
        wait_called = []

        def mock_wait():
            wait_called.append(True)
            original_wait()

        llm._rate_limiter.wait_if_needed = mock_wait

        # Make request
        messages = [HumanMessage(content="Test")]
        result = llm._generate(messages)

        # Verify rate limiter was called
        assert len(wait_called) == 1
        assert result is not None

    @patch("langchain_gradient.chat_models.Gradient")
    def test_rate_limiter_called_during_stream(self, mock_gradient):
        """Test that rate limiter is called during _stream."""
        # Setup mock
        mock_client = Mock()
        mock_gradient.return_value = mock_client

        # Mock streaming response
        mock_chunk1 = Mock()
        mock_chunk1.choices = [Mock()]
        mock_chunk1.choices[0].delta.content = "Hello"

        mock_chunk2 = Mock()
        mock_chunk2.choices = [Mock()]
        mock_chunk2.choices[0].delta.content = " World"

        mock_client.chat.completions.create.return_value = iter(
            [mock_chunk1, mock_chunk2]
        )

        # Create LLM with rate limiting
        llm = ChatGradient(
            api_key="fake-key", enable_rate_limiting=True, max_requests_per_minute=5
        )

        # Spy on rate limiter
        wait_called = []
        original_wait = llm._rate_limiter.wait_if_needed

        def mock_wait():
            wait_called.append(True)
            original_wait()

        llm._rate_limiter.wait_if_needed = mock_wait

        # Make streaming request
        messages = [HumanMessage(content="Test")]
        chunks = list(llm._stream(messages))

        # Verify rate limiter was called
        assert len(wait_called) == 1
        assert len(chunks) == 2

    @patch("langchain_gradient.chat_models.Gradient")
    def test_multiple_requests_increment_usage(self, mock_gradient):
        """Test that multiple requests properly track usage."""
        # Setup mock
        mock_client = Mock()
        mock_gradient.return_value = mock_client

        mock_completion = Mock()
        mock_completion.choices = [Mock()]
        mock_completion.choices[0].message.content = "Response"
        mock_completion.choices[0].finish_reason = "stop"
        mock_completion.usage = Mock(
            prompt_tokens=10, completion_tokens=20, total_tokens=30
        )
        mock_completion.model = "test-model"
        mock_completion.id = "test-id"

        mock_client.chat.completions.create.return_value = mock_completion

        # Create LLM
        llm = ChatGradient(
            api_key="fake-key", enable_rate_limiting=True, max_requests_per_minute=10
        )

        # Make 3 requests
        messages = [HumanMessage(content="Test")]
        for _ in range(3):
            llm._generate(messages)

        # Check usage
        usage = llm._rate_limiter.get_current_usage()
        assert usage["current_requests"] == 3
        assert usage["max_requests"] == 10
        assert usage["usage_percentage"] == 30.0

    def test_rate_limiter_attributes_accessible(self):
        """Test that rate limiter attributes are accessible."""
        llm = ChatGradient(
            api_key="fake-key", enable_rate_limiting=True, max_requests_per_minute=50
        )

        # Should be able to access rate limiter
        assert hasattr(llm, "_rate_limiter")
        assert llm._rate_limiter is not None

        # Should be able to get usage
        usage = llm._rate_limiter.get_current_usage()
        assert isinstance(usage, dict)
        assert "current_requests" in usage
        assert "max_requests" in usage
        assert "usage_percentage" in usage

        # Should be able to reset
        llm._rate_limiter.reset()
        usage_after = llm._rate_limiter.get_current_usage()
        assert usage_after["current_requests"] == 0

    def test_rate_limiting_params_in_model_fields(self):
        """Test that rate limiting params are in ALLOWED_MODEL_FIELDS."""
        from langchain_gradient.constants import ALLOWED_MODEL_FIELDS

        assert "enable_rate_limiting" in ALLOWED_MODEL_FIELDS
        assert "max_requests_per_minute" in ALLOWED_MODEL_FIELDS

    @patch("langchain_gradient.chat_models.Gradient")
    def test_rate_limiting_with_no_api_key_raises_error(self, mock_gradient):
        """Test that missing API key raises error even with rate limiting."""
        llm = ChatGradient(
            api_key=None,  # No API key
            enable_rate_limiting=True,
        )

        with pytest.raises(ValueError, match="Gradient model access key not provided"):
            llm._generate([HumanMessage(content="Test")])

    def test_repr_of_rate_limiter(self):
        """Test string representation of rate limiter."""
        llm = ChatGradient(
            api_key="fake-key", enable_rate_limiting=True, max_requests_per_minute=60
        )

        repr_str = repr(llm._rate_limiter)

        assert "RateLimiter" in repr_str
        assert "max_requests=60" in repr_str
        assert "time_window=60.0s" in repr_str


class TestRateLimiterIntegrationScenarios:
    """Test realistic usage scenarios without API calls."""

    @patch("langchain_gradient.chat_models.Gradient")
    def test_burst_requests_scenario(self, mock_gradient):
        """Simulate burst of requests scenario."""
        mock_client = Mock()
        mock_gradient.return_value = mock_client

        mock_completion = Mock()
        mock_completion.choices = [Mock()]
        mock_completion.choices[0].message.content = "Response"
        mock_completion.choices[0].finish_reason = "stop"
        mock_completion.usage = Mock(
            prompt_tokens=10, completion_tokens=20, total_tokens=30
        )
        mock_completion.model = "test-model"
        mock_completion.id = "test-id"
        mock_client.chat.completions.create.return_value = mock_completion

        llm = ChatGradient(
            api_key="fake-key",
            enable_rate_limiting=True,
            max_requests_per_minute=5,  # Low limit
        )

        # Simulate 10 requests (exceeds limit)
        messages = [HumanMessage(content="Test")]

        # First 5 should be immediate
        for i in range(5):
            result = llm._generate(messages)
            assert result is not None

        # Check we're at limit
        usage = llm._rate_limiter.get_current_usage()
        assert usage["usage_percentage"] == 100.0

    @patch("langchain_gradient.chat_models.Gradient")
    def test_concurrent_requests_scenario(self, mock_gradient):
        """Simulate concurrent requests (thread safety)."""
        from threading import Thread

        mock_client = Mock()
        mock_gradient.return_value = mock_client

        mock_completion = Mock()
        mock_completion.choices = [Mock()]
        mock_completion.choices[0].message.content = "Response"
        mock_completion.choices[0].finish_reason = "stop"
        mock_completion.usage = Mock(
            prompt_tokens=10, completion_tokens=20, total_tokens=30
        )
        mock_completion.model = "test-model"
        mock_completion.id = "test-id"
        mock_client.chat.completions.create.return_value = mock_completion

        llm = ChatGradient(
            api_key="fake-key", enable_rate_limiting=True, max_requests_per_minute=20
        )

        results = []

        def make_request():
            try:
                result = llm._generate([HumanMessage(content="Test")])
                results.append("success")
            except Exception as e:
                results.append(f"error: {e}")

        # Create 10 threads
        threads = [Thread(target=make_request) for _ in range(10)]

        # Start all
        for thread in threads:
            thread.start()

        # Wait for all
        for thread in threads:
            thread.join()

        # All should succeed
        assert len(results) == 10
        assert all(r == "success" for r in results)
