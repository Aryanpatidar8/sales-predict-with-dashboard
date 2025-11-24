import requests

payload = {
    "date": "2023-01-20",
    "store": "StoreA",
    "item": "Item1",
    "promotion": 0,
    "price": 10.0,
}

resp = requests.post("http://127.0.0.1:5000/predict", json=payload, timeout=10)
print("Status:", resp.status_code)
try:
    print("JSON:", resp.json())
except Exception:
    print("Text:", resp.text)
