"""Test timeout and retry functionality for ChatGradient."""
import os
import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from langchain_core.messages import HumanMessage
from langchain_gradient import ChatGradient
from gradient.exceptions import APITimeoutError, APIConnectionError, APIError


# Mock objects for testing
class MockChoice:
    def __init__(self, content="Test response"):
        self.message = Mock(content=content)
        self.delta = Mock(content=content)


class MockCompletion:
    def __init__(self, content="Test response", include_usage=False):
        self.choices = [MockChoice(content)]
        if include_usage:
            self.usage = Mock(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        else:
            self.usage = None


class MockAPIError(Exception):
    """Mock API error with status code."""
    def __init__(self, message="API Error", status_code=None):
        super().__init__(message)
        self.status_code = status_code


@pytest.fixture
def chat_gradient():
    """Create a ChatGradient instance for testing."""
    os.environ["DIGITALOCEAN_INFERENCE_KEY"] = "test-key"
    return ChatGradient(
        model_name="llama3.3-70b-instruct",
        timeout=30.0,
        max_retries=3
    )


class TestTimeoutConfiguration:
    """Test timeout configuration scenarios."""
    
    def test_default_timeout(self):
        """Test that default timeout is set correctly."""
        os.environ["DIGITALOCEAN_INFERENCE_KEY"] = "test-key"
        chat = ChatGradient(model_name="test-model")
        assert chat.timeout == 60.0
    
    def test_custom_timeout(self):
        """Test that custom timeout is set correctly."""
        os.environ["DIGITALOCEAN_INFERENCE_KEY"] = "test-key"
        chat = ChatGradient(model_name="test-model", timeout=30.0)
        assert chat.timeout == 30.0
    
    def test_timeout_passed_to_client(self, chat_gradient):
        """Test that timeout is passed to Gradient client."""
        with patch('langchain_gradient.chat_models.Gradient') as mock_gradient:
            _ = chat_gradient.client
            mock_gradient.assert_called_once()
            call_kwargs = mock_gradient.call_args[1]
            assert 'timeout' in call_kwargs
            assert call_kwargs['timeout'] == 30.0


class TestRetryConfiguration:
    """Test retry configuration scenarios."""
    
    def test_default_max_retries(self):
        """Test that default max_retries is set correctly."""
        os.environ["DIGITALOCEAN_INFERENCE_KEY"] = "test-key"
        chat = ChatGradient(model_name="test-model")
        assert chat.max_retries == 3
    
    def test_custom_max_retries(self):
        """Test that custom max_retries is set correctly."""
        os.environ["DIGITALOCEAN_INFERENCE_KEY"] = "test-key"
        chat = ChatGradient(model_name="test-model", max_retries=5)
        assert chat.max_retries == 5
    
    def test_max_retries_passed_to_client(self, chat_gradient):
        """Test that max_retries is passed to Gradient client."""
        with patch('langchain_gradient.chat_models.Gradient') as mock_gradient:
            _ = chat_gradient.client
            mock_gradient.assert_called_once()
            call_kwargs = mock_gradient.call_args[1]
            assert 'max_retries' in call_kwargs
            assert call_kwargs['max_retries'] == 3


class TestTimeoutHandling:
    """Test timeout handling scenarios."""
    
    def test_timeout_error_raised(self, chat_gradient):
        """Test that APITimeoutError is properly raised after max retries."""
        with patch.object(chat_gradient, 'client') as mock_client:
            mock_client.chat.completions.create.side_effect = APITimeoutError("Request timeout")
            
            messages = [HumanMessage(content="Test")]
            result = chat_gradient._generate(messages)
            
            # Should return error message instead of raising
            assert "ERROR" in result.generations[0].message.content
    
    def test_timeout_retry_attempts(self, chat_gradient):
        """Test that timeout triggers retry attempts."""
        with patch.object(chat_gradient, 'client') as mock_client:
            mock_completion = MockCompletion()
            mock_client.chat.completions.create.side_effect = [
                APITimeoutError("Timeout 1"),
                APITimeoutError("Timeout 2"),
                mock_completion
            ]
            
            messages = [HumanMessage(content="Test")]
            
            # Mock time.sleep to avoid actual delays
            with patch('time.sleep'):
                result = chat_gradient._generate(messages)
            
            # Should succeed on third attempt
            assert result.generations[0].message.content == "Test response"
            assert mock_client.chat.completions.create.call_count == 3


class TestRetryableErrors:
    """Test handling of retryable errors."""
    
    def test_connection_error_retryable(self, chat_gradient):
        """Test that APIConnectionError triggers retries."""
        assert chat_gradient._is_retryable_error(APIConnectionError("Connection failed")) is True
    
    def test_timeout_error_retryable(self, chat_gradient):
        """Test that APITimeoutError triggers retries."""
        assert chat_gradient._is_retryable_error(APITimeoutError("Timeout")) is True
    
    def test_429_rate_limit_retryable(self, chat_gradient):
        """Test that 429 Rate Limit errors trigger retries."""
        error = MockAPIError("Rate limit", status_code=429)
        assert chat_gradient._is_retryable_error(error) is True
    
    def test_500_internal_error_retryable(self, chat_gradient):
        """Test that 500+ errors trigger retries."""
        error = MockAPIError("Internal error", status_code=500)
        assert chat_gradient._is_retryable_error(error) is True
        
        error = MockAPIError("Bad gateway", status_code=502)
        assert chat_gradient._is_retryable_error(error) is True
    
    def test_408_timeout_retryable(self, chat_gradient):
        """Test that 408 Request Timeout errors trigger retries."""
        error = MockAPIError("Request timeout", status_code=408)
        assert chat_gradient._is_retryable_error(error) is True
    
    def test_409_conflict_retryable(self, chat_gradient):
        """Test that 409 Conflict errors trigger retries."""
        error = MockAPIError("Conflict", status_code=409)
        assert chat_gradient._is_retryable_error(error) is True


class TestNonRetryableErrors:
    """Test handling of non-retryable errors."""
    
    def test_400_bad_request_not_retryable(self, chat_gradient):
        """Test that 400 Bad Request errors are not retried."""
        error = MockAPIError("Bad request", status_code=400)
        assert chat_gradient._is_retryable_error(error) is False
    
    def test_401_unauthorized_not_retryable(self, chat_gradient):
        """Test that 401 Unauthorized errors are not retried."""
        error = MockAPIError("Unauthorized", status_code=401)
        assert chat_gradient._is_retryable_error(error) is False
    
    def test_404_not_found_not_retryable(self, chat_gradient):
        """Test that 404 Not Found errors are not retried."""
        error = MockAPIError("Not found", status_code=404)
        assert chat_gradient._is_retryable_error(error) is False


class TestExponentialBackoff:
    """Test exponential backoff retry logic."""
    
    def test_backoff_delay_increases(self, chat_gradient):
        """Test that retry delays increase exponentially."""
        with patch.object(chat_gradient, 'client') as mock_client:
            mock_client.chat.completions.create.side_effect = APITimeoutError("Timeout")
            
            messages = [HumanMessage(content="Test")]
            delays = []
            
            def mock_sleep(delay):
                delays.append(delay)
            
            with patch('time.sleep', side_effect=mock_sleep):
                try:
                    chat_gradient._generate(messages)
                except:
                    pass
            
            # Verify delays increase (with some tolerance for jitter)
            assert len(delays) == 3  # max_retries attempts
            # First delay should be around 2^0 + jitter = 1-2 seconds
            assert 1.0 <= delays[0] <= 2.5
            # Second delay should be around 2^1 + jitter = 2-3 seconds  
            assert 2.0 <= delays[1] <= 3.5
            # Third delay should be around 2^2 + jitter = 4-5 seconds
            assert 4.0 <= delays[2] <= 5.5
    
    def test_backoff_max_delay(self, chat_gradient):
        """Test that backoff delay is capped at 60 seconds."""
        # Set max_retries high to test max delay cap
        chat_gradient.max_retries = 10
        
        with patch.object(chat_gradient, 'client') as mock_client:
            mock_client.chat.completions.create.side_effect = APITimeoutError("Timeout")
            
            messages = [HumanMessage(content="Test")]
            delays = []
            
            def mock_sleep(delay):
                delays.append(delay)
                if len(delays) >= 10:
                    # Stop after 10 attempts
                    raise APITimeoutError("Max retries")
            
            with patch('time.sleep', side_effect=mock_sleep):
                try:
                    chat_gradient._generate(messages)
                except:
                    pass
            
            # Verify no delay exceeds 60 seconds
            for delay in delays:
                assert delay <= 61  # 60 + 1 for jitter


class TestStreamingWithRetry:
    """Test streaming responses with retry logic."""
    
    def test_stream_retry_on_timeout(self, chat_gradient):
        """Test that streaming retries on timeout."""
        with patch.object(chat_gradient, 'client') as mock_client:
            # Create mock stream chunks
            mock_chunks = [
                Mock(choices=[MockChoice("Hello")]),
                Mock(choices=[MockChoice(" World")])
            ]
            
            mock_client.chat.completions.create.side_effect = [
                APITimeoutError("Timeout"),
                iter(mock_chunks)
            ]
            
            messages = [HumanMessage(content="Test")]
            
            with patch('time.sleep'):
                chunks = list(chat_gradient._stream(messages))
            
            # Should succeed on second attempt
            assert len(chunks) == 2
            assert mock_client.chat.completions.create.call_count == 2


class TestErrorMessages:
    """Test error message formatting."""
    
    def test_timeout_error_message(self, chat_gradient):
        """Test that timeout errors include retry information."""
        with patch.object(chat_gradient, 'client') as mock_client:
            mock_client.chat.completions.create.side_effect = APITimeoutError("Original timeout")
            
            messages = [HumanMessage(content="Test")]
            
            with patch('time.sleep'):
                try:
                    # This should raise because all retries are exhausted
                    mock_func = Mock(side_effect=APITimeoutError("Original timeout"))
                    chat_gradient._execute_with_retry(mock_func)
                except APITimeoutError as e:
                    # Error message should mention retries
                    assert "retries" in str(e).lower()


class TestClientInitialization:
    """Test Gradient client initialization."""
    
    def test_client_lazy_initialization(self, chat_gradient):
        """Test that client is initialized lazily."""
        assert chat_gradient._client is None
        _ = chat_gradient.client
        assert chat_gradient._client is not None
    
    def test_client_reuse(self, chat_gradient):
        """Test that client is reused across calls."""
        client1 = chat_gradient.client
        client2 = chat_gradient.client
        assert client1 is client2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
