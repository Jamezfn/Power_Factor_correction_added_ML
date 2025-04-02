import requests
import json

BASE_URL = "http://127.0.0.1:5000"

TEST_INPUT = {
    "input": [
        [[0.1, 0.2, 0.3] for _ in range(50)],
        [[0.4, 0.5, 0.6] for _ in range(50)] 
    ]
}

def test_health():
    response = requests.get(f"{BASE_URL}/health")
    print("/health Response:")
    print(response.status_code, response.json())

def test_predict():
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(f"{BASE_URL}/predict", data=json.dumps(TEST_INPUT), headers=headers)
        print("/predict Response:")
        print(response.status_code)
        if response.ok:
            try:
                print(response.json())
            except json.JSONDecodeError:
                print("Warning: Response is not valid JSON")
                print(response.text)
        else:
            print(f"Error: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to predict endpoint: {e}")

if __name__ == "__main__":
    test_health()
    test_predict()