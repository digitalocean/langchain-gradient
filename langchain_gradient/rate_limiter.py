"""Rate limiting implementation for API requests."""

import time
from collections import deque
from threading import Lock


class RateLimiter:
    """
    Thread-safe rate limiter using token bucket algorithm.

    The token bucket algorithm allows for burst traffic while maintaining
    an average rate limit. Tokens are added at a fixed rate, and each
    request consumes one token.

    Attributes:
        max_requests: Maximum number of requests allowed in the time window
        time_window: Time window in seconds for the rate limit
        requests: Deque storing timestamps of recent requests
        lock: Threading lock for thread-safe operations

    Example:
        >>> limiter = RateLimiter(max_requests=60, time_window=60)
        >>> limiter.wait_if_needed()  # Waits if rate limit would be exceeded
    """

    def __init__(self, max_requests: int = 60, time_window: float = 60.0):
        """
        Initialize rate limiter.

        Args:
            max_requests: Maximum number of requests allowed in time window.
                         Default is 60 requests.
            time_window: Time window in seconds. Default is 60 seconds.

        Raises:
            ValueError: If max_requests or time_window are invalid.
        """
        if max_requests <= 0:
            raise ValueError(f"max_requests must be positive, got {max_requests}")
        if time_window <= 0:
            raise ValueError(f"time_window must be positive, got {time_window}")

        self.max_requests = max_requests
        self.time_window = time_window
        self.requests: deque = deque()
        self.lock = Lock()

    def wait_if_needed(self) -> None:
        """
        Wait if making a request now would exceed the rate limit.

        This method is thread-safe and can be called from multiple threads.
        It automatically removes expired request timestamps and calculates
        the required wait time if the rate limit would be exceeded.

        The method uses a sliding window approach: it only considers requests
        made within the current time window.
        """
        with self.lock:
            current_time = time.time()

            # Remove requests outside the current time window
            while self.requests and self.requests[0] <= current_time - self.time_window:
                self.requests.popleft()

            # Check if we need to wait
            if len(self.requests) >= self.max_requests:
                # Calculate wait time: time until oldest request expires
                oldest_request = self.requests[0]
                wait_time = self.time_window - (current_time - oldest_request)

                if wait_time > 0:
                    print(
                        f"⏳ Rate limit reached ({self.max_requests} requests "
                        f"per {self.time_window}s). Waiting {wait_time:.2f}s..."
                    )
                    time.sleep(wait_time)

                    # Remove the oldest request after waiting
                    self.requests.popleft()

            # Record this request
            self.requests.append(time.time())

    def get_current_usage(self) -> dict:
        """
        Get current rate limit usage statistics.

        Returns:
            Dictionary containing:
                - current_requests: Number of requests in current window
                - max_requests: Maximum allowed requests
                - time_window: Time window in seconds
                - usage_percentage: Percentage of rate limit used
        """
        with self.lock:
            current_time = time.time()

            # Remove expired requests
            while self.requests and self.requests[0] <= current_time - self.time_window:
                self.requests.popleft()

            current_requests = len(self.requests)
            usage_percentage = (current_requests / self.max_requests) * 100

            return {
                "current_requests": current_requests,
                "max_requests": self.max_requests,
                "time_window": self.time_window,
                "usage_percentage": round(usage_percentage, 2),
            }

    def reset(self) -> None:
        """
        Reset the rate limiter by clearing all request history.

        This is useful for testing or when you want to start fresh.
        """
        with self.lock:
            self.requests.clear()

    def __repr__(self) -> str:
        """String representation of the rate limiter."""
        usage = self.get_current_usage()
        return (
            f"RateLimiter("
            f"max_requests={self.max_requests}, "
            f"time_window={self.time_window}s, "
            f"current_usage={usage['current_requests']}/{self.max_requests})"
        )
