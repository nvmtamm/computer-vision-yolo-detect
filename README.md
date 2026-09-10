# VeggieLife - Computer Vision Microservice (YOLOv8)

> **Tính năng**: F-07 / Flow 4: **"Empty the Fridge" AI Chef (USP-1)**  
> **Dự án**: VeggieLife (SWP391 - FPT University)  
> **Thành viên phụ trách**: Huỳnh Việt Phát (Frontend + AI Engineer) & Nguyễn Văn Minh Tâm (Lead Architecture)

Dịch vụ Computer Vision này chịu trách nhiệm nhận diện và bóc tách các loại nguyên liệu thực vật (rau, củ, quả, nấm, đậu hũ) từ ảnh chụp tủ lạnh của người dùng bằng mô hình **YOLOv8**. Kết quả sau đó được chuyển tiếp về Backend ASP.NET Core 8 và Google Gemini 2.5 Flash để tự động gợi ý công thức món chay.

---

## 1. Kiến trúc & Công nghệ Sử dụng

- **Core Engine**: Python 3.11, [FastAPI](https://fastapi.tiangolo.com/), [Ultralytics YOLOv8](https://docs.ultralytics.com/).
- **Mô hình**:
  - *Baseline*: `yolov8n.pt` (Pretrained COCO, 6.2MB) - Nhẹ, nhanh, dùng cho giai đoạn đầu.
  - *Fine-tuned*: `veggie_best.pt` - Huấn luyện với tập dữ liệu rau củ chay đặc thù Việt Nam (đậu phụ, nấm rơm, mướp, v.v.).
- **Containerization**: Docker đa tầng tối ưu CPU-only Torch (< 900MB).
- **Cloud Deployment**: Google Cloud Run (Serverless Container, Cold-start 2-3s, tự động tắt về 0 khi không dùng).

---

## 2. Cài đặt & Chạy Thử trên Máy Local

### Bước 1: Khởi tạo Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate    # Trên Linux / macOS
# Hoặc: .venv\Scripts\activate trên Windows
```

### Bước 2: Cài đặt thư viện
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Bước 3: Khởi chạy Microservice
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

- **Swagger UI trực quan**: Truy cập [http://localhost:8080/docs](http://localhost:8080/docs)
- **Kiểm tra trạng thái**: [http://localhost:8080/health](http://localhost:8080/health)

---

## 3. Chạy Bằng Docker Container

Để kiểm tra môi trường giống hệt Cloud Run:

```bash
# 1. Build image
docker build -t veggielife-cv:latest .

# 2. Chạy container
docker run -p 8080:8080 veggielife-cv:latest
```

---

## 4. Hướng Dẫn Deploy Lên Google Cloud Run (Serverless)

### Cách 1: Deploy trực tiếp bằng Google Cloud SDK (`gcloud`)
```bash
# 1. Đăng nhập GCP và thiết lập Project
gcloud auth login
gcloud config set project [YOUR_PROJECT_ID]

# 2. Build và deploy trực tiếp source code lên Cloud Run
gcloud run deploy veggielife-cv-service \
    --source . \
    --region asia-southeast1 \
    --platform managed \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 1 \
    --min-instances 0 \
    --max-instances 3 \
    --concurrency 10
```

*Lưu ý*: Cloud Run sẽ cung cấp cho nhóm một URL HTTPS công khai dạng:  
`https://veggielife-cv-service-xxxxx-as.a.run.app`  
Chỉ cần dán URL này vào cấu hình `appsettings.json` của Backend ASP.NET Core!

---

## 5. Danh Sách API Endpoints

| Method | Endpoint | Mô tả | Dữ liệu trả về |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | Kiểm tra trạng thái service và model | JSON status |
| `POST` | `/api/v1/cv/detect` | Bóc tách nguyên liệu (dành cho Backend .NET) | JSON Bounding boxes + Danh sách nguyên liệu |
| `POST` | `/api/v1/cv/detect-annotated` | Trả về ảnh có vẽ bounding boxes trực quan | Ảnh JPEG Stream |
| `GET` | `/api/v1/cv/supported-labels` | Danh mục các nhãn và tên tiếng Việt | JSON danh sách nhãn |

---

## 6. Hướng Dẫn Huấn Luyện Fine-Tuning (Cho Bạn Huỳnh Việt Phát)

Để mô hình nhận diện chuẩn các món chay Việt Nam (Đậu phụ, nấm rơm, mướp hương...):
1. Mở file [training/train_colab.ipynb](file:///Users/nguyenvanminhtam/Documents/ComputerVersion/training/train_colab.ipynb) trên Google Colab.
2. Bật GPU miễn phí (T4 GPU).
3. Đăng nhập Roboflow và tải tập dữ liệu gán nhãn `vietnamese-fridge-ingredients`.
4. Chạy lệnh train 50 epochs.
5. Tải file `best.pt` về máy, đổi tên thành `veggie_best.pt` và đặt vào thư mục `weights/`. Service sẽ tự động chuyển sang mô hình mới mà không cần sửa code!

---

## 7. Tích Hợp với ASP.NET Core 8 & Gemini 2.5 Flash

- Mã nguồn C# client đã được viết sẵn tại: [integrations/VeggieLifeVisionClient.cs](file:///Users/nguyenvanminhtam/Documents/ComputerVersion/integrations/VeggieLifeVisionClient.cs).
- Hướng dẫn prompt Gemini 2.5 Flash tại: [integrations/GeminiPromptSample.md](file:///Users/nguyenvanminhtam/Documents/ComputerVersion/integrations/GeminiPromptSample.md).
# computer-vision-yolo-detect
