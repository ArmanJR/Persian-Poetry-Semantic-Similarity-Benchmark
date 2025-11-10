"""
Quick test script to verify OpenRouter API connectivity and response times.
Use this to debug issues before running the full benchmark.
"""

import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/embeddings"

# Test with a simple Persian text
TEST_TEXT = "شب تاریک و بیم موج و گردابی چنین هایل"

# Model to test (change this to the one that's stuck)
TEST_MODEL = "mistralai/mistral-embed-2312"

def test_single_request():
    """Test a single embedding request."""
    if not OPENROUTER_API_KEY:
        print("Error: OPENROUTER_API_KEY not found in .env file")
        return

    print(f"Testing OpenRouter API...")
    print(f"Model: {TEST_MODEL}")
    print(f"Text: {TEST_TEXT[:50]}...")
    print()

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "input": TEST_TEXT,
        "model": TEST_MODEL
    }

    print("Sending request...", flush=True)
    start_time = time.time()

    try:
        response = requests.post(
            OPENROUTER_API_URL,
            headers=headers,
            json=payload,
            timeout=60
        )
        elapsed = time.time() - start_time

        print(f"✓ Response received in {elapsed:.2f}s")
        print(f"Status code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"✓ Success!")
            print(f"Response keys: {list(data.keys())}")

            if 'data' in data and len(data['data']) > 0:
                embedding = data['data'][0]['embedding']
                if isinstance(embedding, list):
                    print(f"Embedding dimension: {len(embedding)}")
                    print(f"First 5 values: {embedding[:5]}")
                elif isinstance(embedding, str):
                    print(f"Embedding is base64 encoded (length: {len(embedding)})")

            if 'usage' in data:
                print(f"Usage: {data['usage']}")

        else:
            print(f"✗ Error: {response.status_code}")
            print(f"Response: {response.text[:500]}")

    except requests.exceptions.Timeout:
        elapsed = time.time() - start_time
        print(f"✗ Request timed out after {elapsed:.2f}s")

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"✗ Error after {elapsed:.2f}s: {e}")


def test_multiple_models():
    """Test multiple models to see which ones work."""
    test_models = [
        "openai/text-embedding-3-small",
        "mistralai/mistral-embed",
        "cohere/embed-multilingual-v3.0",
    ]

    print("Testing multiple models...")
    print("="*60)

    for model in test_models:
        print(f"\nTesting: {model}")
        print("-"*60)

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "input": TEST_TEXT,
            "model": model
        }

        start_time = time.time()

        try:
            response = requests.post(
                OPENROUTER_API_URL,
                headers=headers,
                json=payload,
                timeout=30
            )
            elapsed = time.time() - start_time

            if response.status_code == 200:
                print(f"✓ Success in {elapsed:.2f}s")
            else:
                print(f"✗ Failed: {response.status_code} in {elapsed:.2f}s")

        except requests.exceptions.Timeout:
            print(f"✗ Timeout after 30s")
        except Exception as e:
            print(f"✗ Error: {str(e)[:100]}")

        # Small delay between requests
        time.sleep(0.5)


if __name__ == "__main__":
    print("="*60)
    print("OpenRouter API Test Script")
    print("="*60)
    print()

    # Test single request
    test_single_request()

    # Uncomment to test multiple models:
    # print("\n" + "="*60)
    # test_multiple_models()
