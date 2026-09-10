# Báo Cáo Walkthrough: Hoàn Tất Triển Khai Roboflow Cloud API & Kiến Trúc Lai (Hybrid Vision)

Dự án: **VeggieLife (SWP391 - FPT University)**  
Tính năng: **F-07 / Flow 4: "Empty the Fridge" AI Chef (USP-1 & USP-2)**  
Trạng thái: **Triển khai thành công 100% - 7/7 Unit Tests PASSED**

---

## 1. Các Thay Đổi Kỹ Thuật Đã Thực Hiện (Changes Made)

### 1.1. Cấu Hình & Biến Môi Trường:
- **[.env](file:///Users/nguyenvanminhtam/Documents/ComputerVersion/.env)** & **[.env.example](file:///Users/nguyenvanminhtam/Documents/ComputerVersion/.env.example)**: Thiết lập cấu hình kết nối Roboflow Serverless (`smart-fridge-co7ul-v7nkj/1`), API Key, và biến `INFERENCE_MODE=roboflow`.
- **[app/config.py](file:///Users/nguyenvanminhtam/Documents/ComputerVersion/app/config.py)**: Bổ sung các trường cài đặt Roboflow với Pydantic v2 `SettingsConfigDict`.

### 1.2. Dịch Vụ Roboflow Cloud (`app/services/roboflow_service.py`):
- Xây dựng **`RoboflowService`** sử dụng `httpx.AsyncClient` gọi API Serverless trực tiếp mà không làm nghẽn Event Loop của FastAPI.
- Tự động mã hóa ảnh PIL sang chuẩn Base64 JPEG.
- **Adapter Bounding Box**: Tự động chuyển đổi tọa độ từ hệ tâm $(x, y, w, h)$ của Roboflow sang tọa độ pixel 4 đỉnh $(x_{min}, y_{min}, x_{max}, y_{max})$ của VeggieLife.

### 1.3. Điều Phối Kiến Trúc Lai & Fallback Tự Động (`app/api/routes.py`):
- Cung cấp hàm **`execute_hybrid_detection`**:
  - Ưu tiên gọi model Roboflow Cloud `smart-fridge-co7ul-v7nkj/1`.
  - Nếu gặp sự cố mạng, timeout hoặc hết quota -> **Tự động kích hoạt Fallback sang YOLOv8 Local** và bật cờ `fallback_used: true`, bảo vệ hệ thống không bao giờ bị crash.
  - Cho phép tùy chọn tham số `mode="roboflow"` hoặc `mode="local"` trong mỗi request.

### 1.4. Mở Rộng Từ Điển Song Ngữ (`data/labels_map.json`):
- Bổ sung định nghĩa các class chuyên biệt của tủ lạnh:
  - `Milk` -> *Sữa tươi / Sữa chua* (Gắn cờ cảnh báo chỉ dành cho Lacto-Vegetarian, kiêng với Vegan).
  - `Eggs` / `Egg` -> *Trứng* (Gắn cờ cảnh báo chỉ dành cho Ovo-Vegetarian, kiêng với Vegan).
  - `Bread` -> *Bánh mì* (Tinh bột).
  - `Apple`, `Tomato`...

---

## 2. Kết Quả Kiểm Thử Thực Tế

### 2.1. Kiểm Thử Tự Động (Unit Tests):
Chạy lệnh `pytest`:
```bash
============================= test session starts ==============================
rootdir: /Users/nguyenvanminhtam/Documents/ComputerVersion
configfile: pytest.ini
collected 7 items

tests/test_api.py .......                                                [100%]

============================== 7 passed in 2.37s ===============================
```

### 2.2. Kiểm Thử Live Request Với Ảnh Tủ Lạnh Thực Tế:
Gửi request tới server qua `curl`:
```json
{
  "success": true,
  "execution_time_ms": 2283.4,
  "model_name": "roboflow:smart-fridge-co7ul-v7nkj/1",
  "ingredient_count": 5,
  "detected_ingredients": [
    { "name_en": "Apple", "name_vi": "Táo", "category": "Trái cây", "count": 10, "max_confidence": 0.77 },
    { "name_en": "Eggs", "name_vi": "Trứng", "category": "Trứng gia cầm", "count": 3, "max_confidence": 0.66 },
    { "name_en": "Milk", "name_vi": "Sữa tươi / Sữa chua", "category": "Sữa & Chế phẩm", "count": 1, "max_confidence": 0.64 },
    { "name_en": "Tomato", "name_vi": "Cà chua", "category": "Rau củ", "count": 1, "max_confidence": 0.52 },
    { "name_en": "Bread", "name_vi": "Bánh mì", "category": "Tinh bột / Ngũ cốc", "count": 1, "max_confidence": 0.49 }
  ],
  "fallback_used": false
}
```

Ảnh trực quan vẽ bounding box toàn bộ các vật thể trên đã được tạo thành công tại: [data/samples/roboflow_annotated.jpg](file:///Users/nguyenvanminhtam/Documents/ComputerVersion/data/samples/roboflow_annotated.jpg).
