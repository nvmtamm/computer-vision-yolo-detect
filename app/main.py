import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router as cv_router
from app.config import settings
from app.services.yolo_service import yolo_service

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("veggielife.main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warmup the YOLO model before accepting incoming HTTP requests."""
    logger.info("Khởi động VeggieLife CV Service (Version: %s)...", settings.APP_VERSION)
    logger.info("Đường dẫn model sử dụng: %s", settings.effective_model_path)
    
    # Preload and warm up model
    yolo_service.warmup()
    
    yield
    
    logger.info("Đang tắt dịch vụ VeggieLife CV...")

app = FastAPI(
    title="VeggieLife AI - Computer Vision Service",
    description="""
## Microservice Nhận Diện Nguyên Liệu Tủ Lạnh (F-07 / Flow 4: Empty the Fridge AI Chef)
Dự án SWP391 - VeggieLife (FPT University).

- **Mô hình**: YOLOv8 (Pretrained COCO & Fine-tuned Vietnamese Veggies)
- **Tác vụ**: Phát hiện và bóc tách rau củ, quả, đậu hũ, nấm từ ảnh tủ lạnh.
- **Tích hợp**: Cung cấp API RESTful cho ASP.NET Core 8 Backend và Generative AI (Gemini 2.5 Flash).
    """,
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Setup CORS for Frontend React & Backend .NET Core
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root level health endpoint for GCP Cloud Run
@app.get("/health", tags=["System"])
async def root_health():
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "active_mode": settings.INFERENCE_MODE,
        "roboflow_model": settings.ROBOFLOW_MODEL_ID,
        "local_model": settings.effective_model_path
    }

# Include Computer Vision routes
app.include_router(cv_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8080, reload=True)
