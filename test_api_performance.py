#!/usr/bin/env python3
"""Test API performance and startup times."""

import time

import requests


def test_endpoint(url: str, name: str, timeout: float = 5.0) -> bool:
    """Test a single endpoint and return success status."""
    try:
        start = time.time()
        response = requests.get(url, timeout=timeout)
        end = time.time()
        
        if response.status_code == 200:
            print(f"✅ {name}: {response.status_code} ({end - start:.3f}s)")
            return True
        else:
            print(f"❌ {name}: {response.status_code} ({end - start:.3f}s)")
            return False
    except requests.exceptions.Timeout:
        print(f"⏰ {name}: TIMEOUT after {timeout}s")
        return False
    except Exception as e:
        print(f"❌ {name}: {str(e)}")
        return False

def main():
    """Test API performance."""
    base_url = "http://localhost:8000"
    
    print("🚀 Testing AI-Factory API Performance")
    print("=" * 50)
    
    # Test endpoints in order of importance
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
        success = test_endpoint(url, name)
        results.append((name, success))
        time.sleep(0.1)  # Small delay between requests
    
    print("\n" + "=" * 50)
    print("📊 Summary:")
    
    success_count = sum(1 for _, success in results if success)
    total_count = len(results)
    
    for name, success in results:
        status = "✅" if success else "❌"
        print(f"  {status} {name}")
    
    print(f"\nSuccess Rate: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
    
    if success_count == total_count:
        print("🎉 All endpoints working perfectly!")
    elif success_count >= total_count * 0.8:
        print("⚠️  Most endpoints working, some issues detected.")
    else:
        print("🚨 Major issues detected - check server logs.")

if __name__ == "__main__":
    main()
