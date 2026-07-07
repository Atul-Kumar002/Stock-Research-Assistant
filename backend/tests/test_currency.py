from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_get_exchange_rate():
    response = client.get("/api/exchange-rate")
    assert response.status_code == 200
    data = response.json()
    assert "rate" in data
    assert isinstance(data["rate"], float)
    assert 70.0 < data["rate"] < 100.0  # Assert USD/INR rate is within logical ranges
