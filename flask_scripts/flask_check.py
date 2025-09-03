import requests

# API endpoint
url = "http://127.0.0.1:5000/predict"

# Example wine features (11 values)
data = {
    "features": [8.0, 0.5, 0.46, 2.5, 0.05, 15.0, 46.0, 1.0, 3.3, 0.7, 10.0]
}

# Send POST request
response = requests.post(url, json=data)

print("Status Code:", response.status_code)
print("Response JSON:", response.json())
