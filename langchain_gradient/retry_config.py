import time
import random
from typing import Callable, Optional, Tuple
from dataclasses import dataclass


@dataclass
class RetryConfig:
    """Configuration for retry behavior with exponential backoff."""
    
    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retry_on_exceptions: Tuple = (Exception,)
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt using exponential backoff."""
        delay = min(
            self.initial_delay * (self.exponential_base ** attempt),
            self.max_delay
        )
        
        if self.jitter:
            # Add random jitter (±20% of delay)
            jitter_range = delay * 0.2
            delay += random.uniform(-jitter_range, jitter_range)
        
        return max(0, delay)


def with_retry(
    retry_config: RetryConfig,
    timeout: Optional[float] = None
) -> Callable:
    """Decorator to add retry logic with exponential backoff."""
    
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(retry_config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retry_config.retry_on_exceptions as e:
                    last_exception = e
                    
                    if attempt < retry_config.max_retries:
                        delay = retry_config.calculate_delay(attempt)
                        time.sleep(delay)
                    else:
                        raise APIRetryError(
                            f"Failed after {retry_config.max_retries + 1} attempts",
                            attempts=retry_config.max_retries + 1
                        ) from last_exception
            
            return None
        
        return wrapper
    return decorator
