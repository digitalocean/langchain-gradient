"""Unit tests for RateLimiter class."""

import time
from threading import Thread

import pytest

from langchain_gradient.rate_limiter import RateLimiter


class TestRateLimiterInitialization:
    """Test rate limiter initialization."""

    def test_default_initialization(self):
        """Test rate limiter with default parameters."""
        limiter = RateLimiter()
        assert limiter.max_requests == 60
        assert limiter.time_window == 60.0
        assert len(limiter.requests) == 0

    def test_custom_initialization(self):
        """Test rate limiter with custom parameters."""
        limiter = RateLimiter(max_requests=10, time_window=5.0)
        assert limiter.max_requests == 10
        assert limiter.time_window == 5.0

    def test_invalid_max_requests(self):
        """Test that invalid max_requests raises ValueError."""
        with pytest.raises(ValueError, match="max_requests must be positive"):
            RateLimiter(max_requests=0)

        with pytest.raises(ValueError, match="max_requests must be positive"):
            RateLimiter(max_requests=-1)

    def test_invalid_time_window(self):
        """Test that invalid time_window raises ValueError."""
        with pytest.raises(ValueError, match="time_window must be positive"):
            RateLimiter(time_window=0)

        with pytest.raises(ValueError, match="time_window must be positive"):
            RateLimiter(time_window=-1)


class TestRateLimiterBasicFunctionality:
    """Test basic rate limiter functionality."""

    def test_single_request(self):
        """Test that a single request doesn't wait."""
        limiter = RateLimiter(max_requests=5, time_window=1.0)
        start_time = time.time()
        limiter.wait_if_needed()
        elapsed = time.time() - start_time

        # Should not wait
        assert elapsed < 0.1
        assert len(limiter.requests) == 1

    def test_requests_under_limit(self):
        """Test that requests under the limit don't wait."""
        limiter = RateLimiter(max_requests=5, time_window=1.0)

        start_time = time.time()
        for _ in range(4):
            limiter.wait_if_needed()
        elapsed = time.time() - start_time

        # Should not wait significantly
        assert elapsed < 0.2
        assert len(limiter.requests) == 4

    def test_requests_at_limit_wait(self):
        """Test that exceeding the limit causes waiting."""
        limiter = RateLimiter(max_requests=3, time_window=1.0)

        # Make 3 requests (at limit)
        for _ in range(3):
            limiter.wait_if_needed()

        # 4th request should wait
        start_time = time.time()
        limiter.wait_if_needed()
        elapsed = time.time() - start_time

        # Should wait approximately 1 second
        assert elapsed >= 0.9  # Allow small margin
        assert len(limiter.requests) == 3  # One removed after wait


class TestRateLimiterSlidingWindow:
    """Test sliding window behavior."""

    def test_old_requests_removed(self):
        """Test that old requests are removed from tracking."""
        limiter = RateLimiter(max_requests=5, time_window=0.5)

        # Make 3 requests
        for _ in range(3):
            limiter.wait_if_needed()

        # Wait for window to expire
        time.sleep(0.6)

        # Make another request - should not wait
        start_time = time.time()
        limiter.wait_if_needed()
        elapsed = time.time() - start_time

        assert elapsed < 0.1  # Should not wait
        assert len(limiter.requests) == 1  # Old requests removed

    def test_partial_window_expiry(self):
        """Test behavior when some requests have expired."""
        limiter = RateLimiter(max_requests=3, time_window=1.0)

        # Make 2 requests
        for _ in range(2):
            limiter.wait_if_needed()

        # Wait for half the window
        time.sleep(0.5)

        # Make 2 more requests - total 4, but 2 are older
        for _ in range(2):
            limiter.wait_if_needed()

        assert len(limiter.requests) <= limiter.max_requests
        assert len(limiter.requests) >= 2  # At least the 2 recent ones


class TestRateLimiterThreadSafety:
    """Test thread safety of rate limiter."""

    def test_concurrent_requests(self):
        """Test that concurrent requests are handled safely."""
        limiter = RateLimiter(max_requests=10, time_window=1.0)
        results = []

        def make_request():
            limiter.wait_if_needed()
            results.append(1)

        # Create 15 threads
        threads = [Thread(target=make_request) for _ in range(15)]

        # Start all threads
        start_time = time.time()
        for thread in threads:
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        elapsed = time.time() - start_time

        # All 15 requests should complete
        assert len(results) == 15

        # Should have waited at least once (15 > 10 limit)
        assert elapsed >= 0.9


class TestRateLimiterUtilityMethods:
    """Test utility methods."""

    def test_get_current_usage_empty(self):
        """Test usage stats with no requests."""
        limiter = RateLimiter(max_requests=10, time_window=1.0)
        usage = limiter.get_current_usage()

        assert usage["current_requests"] == 0
        assert usage["max_requests"] == 10
        assert usage["time_window"] == 1.0
        assert usage["usage_percentage"] == 0.0

    def test_get_current_usage_partial(self):
        """Test usage stats with some requests."""
        limiter = RateLimiter(max_requests=10, time_window=1.0)

        for _ in range(5):
            limiter.wait_if_needed()

        usage = limiter.get_current_usage()

        assert usage["current_requests"] == 5
        assert usage["usage_percentage"] == 50.0

    def test_get_current_usage_removes_old(self):
        """Test that usage stats remove old requests."""
        limiter = RateLimiter(max_requests=10, time_window=0.5)

        for _ in range(5):
            limiter.wait_if_needed()

        time.sleep(0.6)

        usage = limiter.get_current_usage()
        assert usage["current_requests"] == 0

    def test_reset(self):
        """Test reset functionality."""
        limiter = RateLimiter(max_requests=10, time_window=1.0)

        for _ in range(5):
            limiter.wait_if_needed()

        assert len(limiter.requests) == 5

        limiter.reset()

        assert len(limiter.requests) == 0

    def test_repr(self):
        """Test string representation."""
        limiter = RateLimiter(max_requests=10, time_window=60.0)

        for _ in range(3):
            limiter.wait_if_needed()

        repr_str = repr(limiter)

        assert "RateLimiter" in repr_str
        assert "max_requests=10" in repr_str
        assert "time_window=60.0s" in repr_str
        assert "current_usage=3/10" in repr_str


class TestRateLimiterEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_high_limit(self):
        """Test with very high request limit."""
        limiter = RateLimiter(max_requests=10000, time_window=1.0)

        start_time = time.time()
        for _ in range(100):
            limiter.wait_if_needed()
        elapsed = time.time() - start_time

        # Should not wait
        assert elapsed < 0.5

    def test_very_short_window(self):
        """Test with very short time window."""
        limiter = RateLimiter(max_requests=2, time_window=0.1)

        # Make 2 requests
        for _ in range(2):
            limiter.wait_if_needed()

        # 3rd should wait
        start_time = time.time()
        limiter.wait_if_needed()
        elapsed = time.time() - start_time

        assert elapsed >= 0.05  # Should wait some time

    def test_single_request_limit(self):
        """Test with limit of 1 request."""
        limiter = RateLimiter(max_requests=1, time_window=0.5)

        limiter.wait_if_needed()

        # 2nd request should wait
        start_time = time.time()
        limiter.wait_if_needed()
        elapsed = time.time() - start_time

        assert elapsed >= 0.4
