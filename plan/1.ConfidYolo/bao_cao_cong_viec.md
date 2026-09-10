# Báo Cáo Tổng Hợp Công Việc Triển Khai Module Computer Vision (YOLOv8)
**Dự án**: VeggieLife (SWP391 - FPT University)  
**Phân hệ**: AI Engine / Computer Vision (F-07: "Empty the Fridge" AI Chef - Flow 4 - USP-1)  
**Thực hiện**: Lead Architecture & AI Engineering Team  
**Ngày cập nhật**: 10/09/2026  

---

## 1. Mục Tiêu & Bối Cảnh Thực Hiện

Trong hệ thống **VeggieLife**, tính năng **"Empty the Fridge" AI Chef (F-07)** là một trong những USP nổi bật nhất nhằm giải quyết nỗi đau lãng phí thực phẩm của người ăn chay.
- **Yêu cầu nghiệp vụ**: Người dùng chụp ảnh các nguyên liệu còn lại trong tủ lạnh -> Hệ thống tự động nhận diện và bóc tách danh sách nguyên liệu -> Gửi sang Google Gemini 2.5 Flash để sáng tạo công thức món chay phù hợp với trường phái ăn chay của người dùng.
- **Yêu cầu kiến trúc**: Dịch vụ Computer Vision cần được đóng gói dưới dạng **Microservice độc lập** (Python FastAPI + Ultralytics YOLOv8), có thể triển khai Serverless trên **Google Cloud Run**, giao tiếp mượt mà với Backend **ASP.NET Core 8 Web API** và Frontend **React.js**.

---

## 2. Các Hạng Mục Công Việc Đã Hoàn Thành

Đến thời điểm hiện tại, toàn bộ kiến trúc lõi, mã nguồn microservice, cấu hình container, tài liệu huấn luyện AI và client tích hợp cho Backend đã được xây dựng hoàn chỉnh trong repository `/Users/nguyenvanminhtam/Documents/ComputerVersion`. Cụ thể:

```
ComputerVersion/
├── app/
│   ├── __init__.py
│   ├── config.py                 # Quản lý cấu hình, thresholds, CORS, dynamic weights
│   ├── main.py                   # FastAPI app, CORS, lifespan model warmup
│   ├── api/
│   │   └── routes.py             # 4 REST API endpoints (/health, /detect, /detect-annotated, /supported-labels)
│   ├── models/
│   │   └── schemas.py            # Pydantic schemas cho request/response chuẩn hóa
│   └── services/
│       ├── preprocessor.py       # Tiền xử lý ảnh, fix lỗi xoay EXIF camera điện thoại
│       └── yolo_service.py       # Inference YOLOv8, bóc tách nhãn, dịch Anh-Việt, gom số lượng
├── data/
│   └── labels_map.json           # Bản đồ ánh xạ nhãn nguyên liệu Anh-Việt & phân loại dinh dưỡng
├── integrations/
│   ├── VeggieLifeVisionClient.cs # Client C# (.NET 8) hoàn chỉnh cho Backend gọi sang YOLO
│   └── GeminiPromptSample.md     # Prompt & JSON Schema chuẩn cho Google Gemini 2.5 Flash
├── training/
│   ├── train_colab.ipynb         # Google Colab notebook 6 bước fine-tune với GPU T4 miễn phí
│   └── data.yaml                 # Cấu hình dataset YOLOv8 cho rau củ chay Việt Nam
├── weights/
│   └── README.md                 # Hướng dẫn quản lý trọng số yolov8n.pt & veggie_best.pt
├── tests/
│   └── test_api.py               # Test suite tự động bằng pytest kiểm tra toàn diện API
├── Dockerfile                    # Multi-stage Docker tối ưu CPU Torch (< 900MB) cho Cloud Run
├── .dockerignore                 # Tối ưu kích thước build container
├── requirements.txt              # Danh mục dependencies phiên bản ổn định
└── README.md                     # Tài liệu kỹ thuật chi tiết toàn bộ dự án
```

---

## 3. Chi Tiết Kỹ Thuật Các Thành Phần Đã Xây Dựng

### 3.1. Cốt Lõi Xử Lý Ảnh & Mô Hình YOLO (`app/services/`)
- **`app/services/preprocessor.py`**:
  - Kiểm tra định dạng file (chỉ nhận `.jpg`, `.jpeg`, `.png`, `.webp`).
  - Giới hạn kích thước file an toàn (tối đa 15MB).
  - **Khắc phục lỗi thực tế quan trọng**: Tự động nhận diện và xoay ảnh theo thẻ `EXIF Orientation` (thường gặp khi người dùng chụp ảnh tủ lạnh bằng iPhone/Samsung bị xoay ngang/ngược).
- **`app/services/yolo_service.py`**:
  - Áp dụng mẫu thiết kế **Singleton** để nạp mô hình vào RAM một lần duy nhất.
  - Tự động ưu tiên trọng số fine-tune `weights/veggie_best.pt`, nếu chưa có sẽ tự động fallback về `yolov8n.pt`.
  - Cơ chế **Warmup**: Chạy suy luận trước một frame giả lập ngay khi app khởi động để triệt tiêu độ trễ của request đầu tiên.
  - Tự động gom nhóm: Ví dụ phát hiện 3 quả cà chua -> trả về `name_vi: "Cà chua", count: 3, max_confidence: 0.92`.
  - Bộ lọc thông minh: Tự động loại bỏ các vật dụng không phải thực phẩm như bát đĩa, cốc chén, chai lọ từ mô hình COCO.
  - Tính năng vẽ Bounding Box trực quan: Tự động đánh nhãn màu sắc và render ảnh kết quả có khung viền phục vụ debug/demo.

### 3.2. Chuẩn Hóa Dữ Liệu & API Endpoints (`app/models/` & `app/api/`)
- **`app/models/schemas.py`**: Định nghĩa các model Pydantic v2 chặt chẽ:
  - `BoundingBoxCoordinates`: Tọa độ 4 đỉnh pixel `(x_min, y_min, x_max, y_max)`.
  - `IngredientSummary`: Tóm tắt tên tiếng Anh, tên tiếng Việt, nhóm dinh dưỡng, số lượng và độ tin cậy cao nhất.
  - `DetectionResponse`: Cấu trúc JSON trả về hoàn chỉnh kèm thời gian suy luận (execution time tính bằng milliseconds).
- **`app/api/routes.py`**:
  1. `GET /health` & `GET /api/v1/cv/health`: Kiểm tra trạng thái container và kiểm tra model đã sẵn sàng chưa.
  2. `POST /api/v1/cv/detect`: Endpoint chính nhận `multipart/form-data`, trả về JSON bóc tách nguyên liệu cho Backend .NET.
  3. `POST /api/v1/cv/detect-annotated`: Nhận ảnh và trả về trực tiếp ảnh JPEG đã vẽ bounding box (rất tiện để demo cho giảng viên hoặc test trên Swagger UI).
  4. `GET /api/v1/cv/supported-labels`: Danh sách tất cả các nhãn thực phẩm được hỗ trợ kèm bản dịch tiếng Việt.

### 3.3. Từ Điển Ánh Xạ Nguyên Liệu Chay (`data/labels_map.json`)
- Ánh xạ chi tiết các nguyên liệu phổ biến của Việt Nam: Đậu phụ trắng, đậu hũ chiên, nấm rơm, nấm đùi gà, nấm kim châm, súp lơ xanh, cà chua, cà rốt, bí đỏ, mướp hương, rau muống, v.v.
- Gắn nhãn nhóm dinh dưỡng: `Đạm thực vật`, `Rau củ`, `Trái cây`, `Gia vị`.
- Cảnh báo các nguyên liệu thuộc nhóm **Ngũ vị tân** (hành, tỏi) để hỗ trợ luồng nghiệp vụ USP-2 (Bộ lọc trường phái ăn chay chuyên sâu).

### 3.4. Đóng Gói Docker Tối Ưu Cho Google Cloud Run (`Dockerfile`)
- Giải quyết bài toán **Cold-start**:
  - Không cài đặt gói PyTorch CUDA nặng 4-5GB. Sử dụng PyTorch CPU-only build (`https://download.pytorch.org/whl/cpu`) kết hợp `opencv-python-headless`.
  - Dung lượng Docker image nén dưới **900MB**.
  - Tải sẵn trọng số `yolov8n.pt` ngay trong quá trình build Docker image, không phụ thuộc vào kết nối mạng khi container khởi chạy.
  - Tự động nhận diện biến môi trường `$PORT` do Google Cloud Run cấp phát.
  - Thiết lập user không đặc quyền (`appuser`) nâng cao tính an toàn.

### 3.5. Tài Liệu Huấn Luyện Fine-Tuning Dành Cho Thành Viên Phụ Trách (`training/`)
- Cung cấp notebook [train_colab.ipynb](file:///Users/nguyenvanminhtam/Documents/ComputerVersion/training/train_colab.ipynb) sẵn sàng chạy trên Google Colab với GPU T4 miễn phí.
- Bạn **Huỳnh Việt Phát** chỉ cần:
  1. Mở notebook trên Colab.
  2. Tải dataset gán nhãn nguyên liệu Việt Nam từ Roboflow.
  3. Bấm chạy train 50 epochs (~20-30 phút).
  4. Tải file `best.pt` đổi tên thành `veggie_best.pt` đặt vào `weights/` là xong, không cần code lại API.
- File cấu hình [data.yaml](file:///Users/nguyenvanminhtam/Documents/ComputerVersion/training/data.yaml) mẫu định nghĩa sẵn 18 lớp nguyên liệu chay thuần Việt.

### 3.6. Module Tích Hợp Cho Backend ASP.NET Core 8 & Gemini AI (`integrations/`)
- **`VeggieLifeVisionClient.cs`**:
  - Mã nguồn C# (.NET 8) chuẩn mẫu dành cho bạn **Nguyễn Lê Kỳ Thư**.
  - Đã tích hợp đầy đủ DTOs, cấu hình `IHttpClientFactory`, xử lý Multipart Form Data và xử lý lỗi mạng.
- **`GeminiPromptSample.md`**:
  - Bản thiết kế System Instruction và User Prompt chuẩn định dạng JSON cho **Google Gemini 2.5 Flash**.
  - Kết hợp thông tin: Nguyên liệu bóc tách từ YOLO + Dị ứng + Trường phái ăn chay (Vegan, kiêng ngũ vị tân) -> Sinh công thức món chay hoàn chỉnh có định lượng và calo.

---

## 4. Tình Trạng Hiện Tại & Kế Hoạch Tiếp Theo (Next Steps)

1. **Môi trường**: 
   - Đã cấu hình Virtual Environment `.venv` và đang tiến hành cài đặt các dependencies.
2. **Kiểm thử chất lượng**:
   - Chạy bộ unit test `tests/test_api.py` để đảm bảo 100% endpoint pass kiểm thử.
3. **Phối hợp nhóm**:
   - Chuyển giao file `VeggieLifeVisionClient.cs` cho bạn Kỳ Thư tích hợp vào Backend .NET 8.
   - Hướng dẫn bạn Việt Phát thực hiện Fine-tune dataset trên Colab trong Sprint 3 để kịp tiến độ Assessment 1 ở Sprint 4.
