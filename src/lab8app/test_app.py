import requests
import json

# Test data
test_data = {
    "text": "This is a test Reddit post about artificial intelligence and machine learning"
}

# Make request to the local API
response = requests.post(
    "http://localhost:8000/predict",
    json=test_data
)

print("Request data:", json.dumps(test_data, indent=2))
print("\nResponse:", json.dumps(response.json(), indent=2))