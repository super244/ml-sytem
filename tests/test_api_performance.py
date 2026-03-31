#!/usr/bin/env python3
"""Test API performance and startup times."""

import time

import requests


def check_endpoint(url: str, name: str, timeout: float = 5.0) -> bool:
    """Test a single endpoint and return success status."""
    try:
        start = time.time()
        response = requests.get(url, timeout=timeout)
        end = time.time()

        if response.status_code == 200:
            print(f"OK {name}: {response.status_code} ({end - start:.3f}s)")
            return True
        else:
            print(f"FAIL {name}: {response.status_code} ({end - start:.3f}s)")
            return False
    except requests.exceptions.Timeout:
        print(f"TIMEOUT {name}: TIMEOUT after {timeout}s")
        return False
    except Exception as e:
        print(f"ERROR {name}: {str(e)}")
        return False


def test_check_endpoint_returns_bool() -> None:
    """Verify check_endpoint returns a bool for unreachable hosts."""
    result = check_endpoint("http://127.0.0.1:19999/health", "unreachable", timeout=0.5)
    assert isinstance(result, bool)
    assert result is False


def main():
    """Test API performance."""
    base_url = "http://localhost:8000"

    print("AI-Factory API Performance Test")
    print("=" * 50)

    endpoints = [
        ("/health", "Health Check"),
        ("/status", "Status"),
        ("/workspace/fast", "Fast Workspace"),
        ("/workspace/status", "Workspace Status"),
        ("/v1/models", "Models"),
        ("/v1/workspace", "Full Workspace"),
    ]

    results = []
    for endpoint, name in endpoints:
        url = f"{base_url}{endpoint}"
        success = check_endpoint(url, name)
        results.append((name, success))
        time.sleep(0.1)

    print("\n" + "=" * 50)
    print("Summary:")

    success_count = sum(1 for _, success in results if success)
    total_count = len(results)

    for name, success in results:
        status = "OK" if success else "FAIL"
        print(f"  {status} {name}")

    print(f"\nSuccess Rate: {success_count}/{total_count} ({success_count / total_count * 100:.1f}%)")

    if success_count == total_count:
        print("All endpoints working perfectly!")
    elif success_count >= total_count * 0.8:
        print("Most endpoints working, some issues detected.")
    else:
        print("Major issues detected - check server logs.")


if __name__ == "__main__":
    main()
