import pytest
import time
from unittest.mock import Mock, patch
import httpx
from langchain_gradient.chat_models import ChatGradient
from langchain_gradient.exceptions import APITimeoutError, APIRetryError
from langchain_gradient.retry_config import RetryConfig


class TestTimeoutHandling:
    """Test timeout configuration and error handling."""
    
    def test_default_timeout_configuration(self):
        """Test default timeout is set correctly."""
        llm = ChatGradient(model="llama3.3-70b-instruct")
        assert llm.timeout == 30.0
    
    def test_custom_timeout_configuration(self):
        """Test custom timeout can be configured."""
        llm = ChatGradient(
            model="llama3.3-70b-instruct",
            timeout=60.0
        )
        assert llm.timeout == 60.0
    
    @patch('httpx.Client.post')
    def test_timeout_exception_raised(self, mock_post):
        """Test APITimeoutError is raised on timeout."""
        mock_post.side_effect = httpx.TimeoutException("Connection timeout")
        
        llm = ChatGradient(
            model="llama3.3-70b-instruct",
            timeout=5.0,
            max_retries=0
        )
        
        with pytest.raises(APITimeoutError) as exc_info:
            llm.invoke("Hello world!")
        
        assert "timed out" in str(exc_info.value)
        assert exc_info.value.timeout == 5.0


class TestRetryLogic:
    """Test retry logic with exponential backoff."""
    
    def test_exponential_backoff_calculation(self):
        """Test exponential backoff delay calculation."""
        config = RetryConfig(
            initial_delay=1.0,
            exponential_base=2.0,
            jitter=False
        )
        
        # Without jitter, delays should be: 1, 2, 4, 8...
        assert config.calculate_delay(0) == 1.0
        assert config.calculate_delay(1) == 2.0
        assert config.calculate_delay(2) == 4.0
        assert config.calculate_delay(3) == 8.0
    
    def test_max_delay_cap(self):
        """Test maximum delay is capped correctly."""
        config = RetryConfig(
            initial_delay=1.0,
            max_delay=10.0,
            exponential_base=2.0,
            jitter=False
        )
        
        # Delay should be capped at max_delay
        assert config.calculate_delay(10) == 10.0
    
    @patch('httpx.Client.post')
    def test_retry_on_server_error(self, mock_post):
        """Test retries are attempted on 5xx errors."""
        # Simulate 2 failures then success
        mock_post.side_effect = [
            Mock(status_code=503, raise_for_status=Mock(
                side_effect=httpx.HTTPStatusError(
                    "Service Unavailable",
                    request=Mock(),
                    response=Mock(status_code=503)
                )
            )),
            Mock(status_code=503, raise_for_status=Mock(
                side_effect=httpx.HTTPStatusError(
                    "Service Unavailable",
                    request=Mock(),
                    response=Mock(status_code=503)
                )
            )),
            Mock(
                status_code=200,
                json=lambda: {
                    "choices": [{
                        "message": {"content": "Success!"}
                    }]
                }
            )
        ]
        
        llm = ChatGradient(
            model="llama3.3-70b-instruct",
            max_retries=3
        )
        
        response = llm.invoke("Hello world!")
        assert response == "Success!"
        assert mock_post.call_count == 3
    
    @patch('httpx.Client.post')
    def test_max_retries_exceeded(self, mock_post):
        """Test APIRetryError raised when max retries exceeded."""
        mock_post.side_effect = httpx.TimeoutException("Timeout")
        
        llm = ChatGradient(
            model="llama3.3-70b-instruct",
            max_retries=2,
            timeout=1.0
        )
        
        with pytest.raises(APIRetryError) as exc_info:
            llm.invoke("Hello world!")
        
        assert "3 attempts" in str(exc_info.value)
        assert mock_post.call_count == 3  # Initial + 2 retries


class TestCustomRetryConfiguration:
    """Test custom retry configuration options."""
    
    def test_custom_retry_config(self):
        """Test custom RetryConfig can be provided."""
        custom_config = RetryConfig(
            max_retries=5,
            initial_delay=2.0,
            max_delay=120.0,
            exponential_base=3.0,
            jitter=False
        )
        
        llm = ChatGradient(
            model="llama3.3-70b-instruct",
            retry_config=custom_config
        )
        
        assert llm.retry_config.max_retries == 5
        assert llm.retry_config.initial_delay == 2.0
        assert llm.retry_config.exponential_base == 3.0
