import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import settings
from app.main import app

client = TestClient(app)

def create_sample_image(format="JPEG") -> bytes:
    """Create a simple in-memory RGB image for testing."""
    img = Image.new("RGB", (200, 200), color=(34, 139, 34))
    buffer = io.BytesIO()
    img.save(buffer, format=format)
    return buffer.getvalue()

def test_root_health():
    """Verify root health endpoint returns healthy status and active mode."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "VeggieLife" in data["service"]
    assert "active_mode" in data

def test_api_health():
    """Verify CV API health endpoint returns status ok and mode info."""
    response = client.get("/api/v1/cv/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_path" in data
    assert "active_mode" in data

def test_supported_labels():
    """Verify supported labels endpoint returns food labels including smart fridge classes."""
    response = client.get("/api/v1/cv/supported-labels")
    assert response.status_code == 200
    data = response.json()
    assert data["total_labels"] > 0
    names = [lbl["name_en"].lower() for lbl in data["labels"]]
    assert "apple" in names
    assert "milk" in names
    assert "eggs" in names
    assert "bread" in names

def test_detect_with_invalid_file_type():
    """Verify invalid file extensions are rejected with 400 Bad Request."""
    response = client.post(
        "/api/v1/cv/detect",
        files={"file": ("malicious.exe", b"binary content", "application/octet-stream")}
    )
    assert response.status_code == 400
    assert "không được hỗ trợ" in response.json()["detail"]

def test_detect_roboflow_cloud():
    """Verify direct Roboflow Serverless Cloud API call."""
    img_bytes = create_sample_image()
    response = client.post(
        "/api/v1/cv/detect",
        files={"file": ("fridge_test.jpg", img_bytes, "image/jpeg")},
        data={"confidence": 0.35}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "execution_time_ms" in data
    assert "detected_ingredients" in data
    assert "roboflow" in data["model_name"]

def test_detect_annotated_image():
    """Verify annotated image endpoint returns a valid JPEG stream with watermark."""
    img_bytes = create_sample_image()
    response = client.post(
        "/api/v1/cv/detect-annotated",
        files={"file": ("fridge_test.jpg", img_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert "roboflow" in response.headers["x-model-used"]
    assert len(response.content) > 0
