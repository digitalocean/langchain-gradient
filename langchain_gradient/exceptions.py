class APITimeoutError(Exception):
    """Exception raised when API request times out."""
    
    def __init__(self, message: str = "API request timed out", timeout: float = None):
        self.timeout = timeout
        self.message = f"{message} (timeout={timeout}s)" if timeout else message
        super().__init__(self.message)


class APIRetryError(Exception):
    """Exception raised when max retries are exceeded."""
    
    def __init__(self, message: str = "Max retries exceeded", attempts: int = None):
        self.attempts = attempts
        self.message = f"{message} (attempts={attempts})" if attempts else message
        super().__init__(self.message)
