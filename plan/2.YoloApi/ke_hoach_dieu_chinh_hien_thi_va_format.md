# Kế Hoạch Điều Chỉnh: Giữ Nguyên Format Tiếng Anh Từ Cloud & Tối Ưu Hiển Thị Bounding Box

**Dự án**: VeggieLife (SWP391 - FPT University)  
**Phân hệ**: AI Engine / Computer Vision (F-07: "Empty the Fridge" AI Chef)  
**Thực hiện**: Huỳnh Việt Phát (AI Engineer) & Nguyễn Văn Minh Tâm (Lead Architecture)  
**Ngày lập**: 10/09/2026  

---

## 1. Phân Tích Các Vấn Đề Hiện Tại Từ Thực Tế

Dựa trên kết quả chạy thử nghiệm trực tiếp trên giao diện Swagger UI:

1. **Vấn đề 1: Nhãn bị tự động chuyển ngữ sang tiếng Việt & sai format gốc**:
   - Hiện tại hệ thống đang cố gắng dịch class sang tiếng Việt qua file từ điển (`labels_map.json`).
   - Yêu cầu mới: **Giữ nguyên 100% format tiếng Anh chuẩn do Cloud (Roboflow / YOLO) gửi về** (`Apple`, `Milk`, `Eggs`, `Tomato`, `Bread`, `Carrot`...). Không tự ý convert sang tiếng Việt làm thay đổi dữ liệu gốc của model.
   - Giữ nguyên các trường nguyên bản của Roboflow: `class`, `confidence`, `x`, `y`, `width`, `height`, `class_id`, `detection_id`.

2. **Vấn đề 2: Đường viền Bounding Box quá mảnh**:
   - Hiện tại độ dày viền cố định là `width = 3` pixel.
   - Đối với ảnh chụp tủ lạnh có độ phân giải cao (ví dụ `1920 x 2089` như trong ảnh test), nét vẽ 3px chỉ chiếm **0.14%** khung hình, khiến đường viền mỏng dính như sợi chỉ, rất khó quan sát.

3. **Vấn đề 3: Cỡ chữ (font size) tên Class quá nhỏ, không thể đọc được**:
   - Hiện tại hệ thống dùng font bitmap mặc định không cấu hình kích cỡ (cố định ~10px).
   - Trên ảnh độ phân giải lớn, nhãn chữ bị thu nhỏ thành một vệt trắng li ti không thể đọc được chữ gì.

4. **Vấn đề 4: Model đang chạy fallback YOLOv8 thay vì Roboflow**:
   - Trong response header của ảnh chụp màn hình: `x-model-used: yolov8n.pt`.
   - Nguyên nhân: Tiến trình `uvicorn` được bật từ trước khi tạo file `.env`, nên chưa nạp biến `INFERENCE_MODE=roboflow`.

---

## 2. Giải Pháp Kỹ Thuật Chi Tiết

### 2.1. Chuẩn Hóa Format Dữ Liệu Tiếng Anh (Pure English Cloud Format)
- Giữ nguyên nhãn gốc: `class: "Apple"`, `confidence: 0.7713`.
- Bỏ cơ chế ép buộc dịch sang tiếng Việt. Nếu có tiếng Việt thì chỉ lưu dưới dạng trường phụ `name_vi` tùy chọn, còn trường định danh chính phải là `class` hoặc `name` tiếng Anh.
- Response Schema chuẩn hóa:
  ```json
  {
    "success": true,
    "execution_time_ms": 85.2,
    "model_name": "roboflow:smart-fridge-co7ul-v7nkj/1",
    "ingredient_count": 5,
    "detected_ingredients": [
      {
        "name": "Apple",
        "count": 10,
        "max_confidence": 0.7713
      },
      {
        "name": "Eggs",
        "count": 3,
        "max_confidence": 0.6584
      }
    ],
    "predictions": [
      {
        "class": "Apple",
        "confidence": 0.7713,
        "x": 548.5,
        "y": 1582.5,
        "width": 169.0,
        "height": 171.0,
        "box": {
          "x_min": 464,
          "y_min": 1497,
          "x_max": 633,
          "y_max": 1668
        }
      }
    ]
  }
  ```

---

### 2.2. Tính Độ Dày Viền Động (Dynamic Bounding Box Thickness)
Thay vì dùng nét vẽ cố định 3px, độ dày nét vẽ sẽ tự động co giãn theo kích thước ảnh thực tế:
$$\text{line\_thickness} = \max\left(4, \text{int}\left(\frac{\min(\text{width}, \text{height})}{250}\right)\right)$$

- Với ảnh nhỏ ($640 \times 480$): nét vẽ $\approx 4\text{px}$.
- Với ảnh lớn ($1920 \times 2089$): nét vẽ $\approx 8\text{px}$ (dày gấp gần 3 lần hiện tại, cực kỳ rõ nét).

---

### 2.3. Tính Cỡ Chữ Động & Render Badge Nổi Bật (Dynamic Font Scaling & Badging)
1. **Kích thước chữ co giãn tự động**:
   $$\text{font\_size} = \max\left(20, \text{int}\left(\frac{\min(\text{width}, \text{height})}{45}\right)\right)$$
   - Với ảnh $1920 \times 2089$: Cỡ chữ sẽ đạt **$\approx 42\text{px}$** (gấp hơn 4 lần hiện tại)!
2. **Cơ chế nạp Font sắc nét**:
   - Sử dụng `ImageFont.truetype(...)` với các font chuẩn có sẵn trên macOS / Linux (`Helvetica`, `Arial`, hoặc `ImageFont.load_default(size=font_size)`).
3. **Badge nhãn chữ**:
   - Nội dung nhãn: `f"{item.label} {int(item.confidence * 100)}%"` (Ví dụ: `Apple 77%`, `Milk 64%`).
   - Vẽ khung badge nền màu đậm tương ứng với Bounding Box, padding trên dưới 6px, trái phải 10px.
   - Màu chữ: Trắng tinh (`#FFFFFF`), tương phản mạnh trên nền màu sắc.

---

## 3. Các Bước Thực Hiện Cụ Thể (Action Plan)

1. **Bước 1: Cập nhật hàm `draw_annotations` trong [app/services/yolo_service.py](file:///Users/nguyenvanminhtam/Documents/ComputerVersion/app/services/yolo_service.py)**:
   - Thêm thuật toán tính toán `line_thickness` và `font_size` động theo kích thước ảnh `(image.width, image.height)`.
   - Nạp font TrueType / scalable font kích thước lớn.
   - Hiển thị nhãn bằng tiếng Anh gốc (`item.label`).

2. **Bước 2: Chuẩn hóa Schema trong [app/models/schemas.py](file:///Users/nguyenvanminhtam/Documents/ComputerVersion/app/models/schemas.py)**:
   - Đặt trường `class_name` hoặc `name` tiếng Anh làm gốc.
   - Thêm trường `predictions` bảo lưu nguyên vẹn định dạng Cloud Roboflow `(x, y, width, height, confidence, class)`.

3. **Bước 3: Cập nhật [app/services/roboflow_service.py](file:///Users/nguyenvanminhtam/Documents/ComputerVersion/app/services/roboflow_service.py)**:
   - Giữ nguyên nhãn gốc không ép chuyển đổi tiếng Việt.
   - Lưu trữ cả tọa độ tâm Cloud và tọa độ 4 góc pixel.

4. **Bước 4: Cập nhật [app/api/routes.py](file:///Users/nguyenvanminhtam/Documents/ComputerVersion/app/api/routes.py)**:
   - Đảm bảo endpoint `/detect-annotated` áp dụng hàm vẽ mới với nhãn tiếng Anh rõ nét.

5. **Bước 5: Kiểm tra và Render lại ảnh mẫu**:
   - Chạy test lại với `data/samples/fridge_sample.jpg`.
   - Lưu ảnh kết quả mới vào `data/samples/roboflow_annotated.jpg` và kiểm tra độ sắc nét của chữ và đường viền.
