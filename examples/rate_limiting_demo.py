"""
Demonstration of rate limiting functionality.

This example shows how rate limiting prevents API throttling errors
when making many rapid requests.
"""

import os
import time

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from langchain_gradient import ChatGradient

load_dotenv()


def demo_without_rate_limiting():
    """
    Demo: Making requests WITHOUT rate limiting.

    This will likely fail with 429 errors if you exceed API limits.
    """
    print("=" * 60)
    print("DEMO 1: Without Rate Limiting (May Fail)")
    print("=" * 60)

    llm = ChatGradient(
        model="llama3.3-70b-instruct",
        api_key=os.getenv("DIGITALOCEAN_INFERENCE_KEY"),
        enable_rate_limiting=False,  # Disabled
    )

    print("Making 10 rapid requests...")
    start_time = time.time()

    success_count = 0
    error_count = 0

    for i in range(10):
        try:
            response = llm.invoke([HumanMessage(content=f"Say 'Request {i}'")])
            success_count += 1
            print(f"✅ Request {i}: Success")
        except Exception as e:
            error_count += 1
            print(f"❌ Request {i}: Failed - {str(e)[:50]}")

    elapsed = time.time() - start_time

    print("\nResults:")
    print(f"  Time taken: {elapsed:.2f}s")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {error_count}")
    print()


def demo_with_rate_limiting():
    """
    Demo: Making requests WITH rate limiting.

    This will succeed for all requests, automatically waiting when needed.
    """
    print("=" * 60)
    print("DEMO 2: With Rate Limiting (All Succeed)")
    print("=" * 60)

    llm = ChatGradient(
        model="llama3.3-70b-instruct",
        api_key=os.getenv("DIGITALOCEAN_INFERENCE_KEY"),
        enable_rate_limiting=True,
        max_requests_per_minute=5,  # Low limit for demo purposes
    )

    print("Making 10 requests with rate limiting...")
    print(f"Rate limit: {llm.max_requests_per_minute} requests/minute\n")

    start_time = time.time()

    for i in range(10):
        # Check current usage before request
        usage = llm._rate_limiter.get_current_usage()
        print(f"Request {i}: Current usage = {usage['usage_percentage']:.1f}%")

        response = llm.invoke([HumanMessage(content=f"Say 'Request {i}'")])
        print(f"✅ Request {i}: Success - {response.content[:30]}...")

    elapsed = time.time() - start_time

    print("\nResults:")
    print(f"  Time taken: {elapsed:.2f}s")
    print("  All requests succeeded!")
    print("  Rate limiting prevented any failures")
    print()


def demo_usage_monitoring():
    """
    Demo: Monitor rate limit usage in real-time.
    """
    print("=" * 60)
    print("DEMO 3: Real-Time Usage Monitoring")
    print("=" * 60)

    llm = ChatGradient(
        model="llama3.3-70b-instruct",
        api_key=os.getenv("DIGITALOCEAN_INFERENCE_KEY"),
        enable_rate_limiting=True,
        max_requests_per_minute=10,
    )

    print("Making 5 requests and monitoring usage...\n")

    for i in range(5):
        # Get usage BEFORE request
        usage_before = llm._rate_limiter.get_current_usage()

        response = llm.invoke([HumanMessage(content=f"Request {i}")])

        # Get usage AFTER request
        usage_after = llm._rate_limiter.get_current_usage()

        print(f"Request {i}:")
        print(
            f"  Before: {usage_before['current_requests']}/{usage_before['max_requests']} "
            f"({usage_before['usage_percentage']:.1f}%)"
        )
        print(
            f"  After:  {usage_after['current_requests']}/{usage_after['max_requests']} "
            f"({usage_after['usage_percentage']:.1f}%)"
        )
        print()

    print("Final usage stats:")
    final_usage = llm._rate_limiter.get_current_usage()
    print(f"  Current requests in window: {final_usage['current_requests']}")
    print(f"  Maximum allowed: {final_usage['max_requests']}")
    print(f"  Usage percentage: {final_usage['usage_percentage']:.1f}%")
    print()


def demo_custom_limits():
    """
    Demo: Using custom rate limits for different scenarios.
    """
    print("=" * 60)
    print("DEMO 4: Custom Rate Limits")
    print("=" * 60)

    scenarios = [
        ("Conservative (30/min)", 30),
        ("Standard (60/min)", 60),
        ("Aggressive (120/min)", 120),
    ]

    for name, limit in scenarios:
        print(f"\nScenario: {name}")

        llm = ChatGradient(
            model="llama3.3-70b-instruct",
            api_key=os.getenv("DIGITALOCEAN_INFERENCE_KEY"),
            enable_rate_limiting=True,
            max_requests_per_minute=limit,
        )

        print(f"  Rate limit: {limit} requests/minute")
        print(f"  That's 1 request every {60 / limit:.2f} seconds")

        # Make 3 quick requests
        start = time.time()
        for i in range(3):
            llm.invoke([HumanMessage(content="Quick test")])
        elapsed = time.time() - start

        print(f"  Time for 3 requests: {elapsed:.2f}s")

    print()


def demo_rate_limiter_reset():
    """
    Demo: Resetting the rate limiter.
    """
    print("=" * 60)
    print("DEMO 5: Rate Limiter Reset")
    print("=" * 60)

    llm = ChatGradient(
        model="llama3.3-70b-instruct",
        api_key=os.getenv("DIGITALOCEAN_INFERENCE_KEY"),
        enable_rate_limiting=True,
        max_requests_per_minute=5,
    )

    print("Making 5 requests to hit the limit...")
    for i in range(5):
        llm.invoke([HumanMessage(content=f"Request {i}")])

    usage = llm._rate_limiter.get_current_usage()
    print(
        f"Usage after 5 requests: {usage['current_requests']}/{usage['max_requests']}"
    )

    print("\nResetting rate limiter...")
    llm._rate_limiter.reset()

    usage_after = llm._rate_limiter.get_current_usage()
    print(
        f"Usage after reset: {usage_after['current_requests']}/{usage_after['max_requests']}"
    )

    print("\nRate limiter is now cleared and ready for new requests!")
    print()


def demo_streaming_with_rate_limiting():
    """
    Demo: Rate limiting with streaming responses.
    """
    print("=" * 60)
    print("DEMO 6: Streaming with Rate Limiting")
    print("=" * 60)

    llm = ChatGradient(
        model="llama3.3-70b-instruct",
        api_key=os.getenv("DIGITALOCEAN_INFERENCE_KEY"),
        enable_rate_limiting=True,
        max_requests_per_minute=5,
        streaming=True,
    )

    print("Making 3 streaming requests...\n")

    for i in range(3):
        print(f"Stream {i}: ", end="", flush=True)

        for chunk in llm.stream([HumanMessage(content=f"Count to 3 for request {i}")]):
            print(chunk.content, end="", flush=True)

        print()  # New line after stream

        usage = llm._rate_limiter.get_current_usage()
        print(f"  Usage: {usage['usage_percentage']:.1f}%\n")

    print("All streaming requests completed successfully!")
    print()


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("DIGITALOCEAN_INFERENCE_KEY"):
        print("❌ Error: DIGITALOCEAN_INFERENCE_KEY not set!")
        print("Please set your API key in .env file")
        exit(1)

    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "RATE LIMITING DEMONSTRATIONS" + " " * 20 + "║")
    print("╚" + "═" * 58 + "╝")
    print("\n")

    # Run demos
    try:
        # Demo 1: Without rate limiting (may fail)
        # Uncomment at your own risk - may cause 429 errors!
        # demo_without_rate_limiting()

        # Demo 2: With rate limiting (always succeeds)
        demo_with_rate_limiting()

        # Demo 3: Usage monitoring
        demo_usage_monitoring()

        # Demo 4: Custom limits
        demo_custom_limits()

        # Demo 5: Reset functionality
        demo_rate_limiter_reset()

        # Demo 6: Streaming
        demo_streaming_with_rate_limiting()

        print("=" * 60)
        print("All demos completed successfully! ✅")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Error during demo: {str(e)}")
        raise
