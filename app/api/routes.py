import io
import logging
from typing import Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import Response
from PIL import Image

from app.config import settings
from app.models.schemas import (
    DetectionResponse,
    HealthResponse,
    LabelInfo,
    SupportedLabelsResponse,
)
from app.services.preprocessor import load_and_preprocess_image
from app.services.roboflow_service import roboflow_service
from app.services.yolo_service import yolo_service

logger = logging.getLogger("veggielife.routes")
router = APIRouter(prefix="/api/v1/cv", tags=["Computer Vision - Fridge Ingredients"])

@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Kiểm tra tình trạng hoạt động của Roboflow Cloud CV microservice."""
    return HealthResponse(
        status="ok",
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        active_mode="roboflow_cloud_only",
        roboflow_model=settings.ROBOFLOW_MODEL_ID,
        model_loaded=True,
        model_path=f"roboflow:{settings.ROBOFLOW_MODEL_ID}"
    )

@router.post(
    "/detect",
    response_model=DetectionResponse,
    summary="Bóc tách nguyên liệu từ ảnh tủ lạnh (100% Roboflow Serverless Cloud)",
    description="Gửi ảnh tới mô hình Roboflow Cloud (smart-fridge-co7ul-v7nkj/1) và trả về nguyên liệu cùng predictions."
)
async def detect_fridge_ingredients(
    file: UploadFile = File(..., description="File ảnh tủ lạnh (jpg, png, webp)"),
    confidence: Optional[float] = Form(
        None,
        ge=0.1,
        le=1.0,
        description="Ngưỡng tin cậy tối thiểu (mặc định 0.40)"
    ),
    filter_non_food: bool = Form(
        True,
        description="Tự động lọc bỏ các vật thể không phải thực phẩm"
    )
):
    image = await load_and_preprocess_image(file)
    
    try:
        response = await roboflow_service.detect_ingredients(
            image=image,
            conf_threshold=confidence,
            filter_non_food=filter_non_food
        )
        return response
    except Exception as e:
        logger.error("Lỗi khi gọi Roboflow Serverless Cloud: %s", e)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Lỗi từ dịch vụ Roboflow Serverless Cloud: {str(e)}"
        )

@router.post(
    "/detect-annotated",
    summary="Nhận diện qua Roboflow và trả về ảnh có vẽ Bounding Boxes",
    description="Nhận diện trực tiếp từ Cloud và vẽ bounding boxes sắc nét kèm đóng dấu watermark Roboflow Cloud."
)
async def detect_and_draw_boxes(
    file: UploadFile = File(..., description="File ảnh tủ lạnh"),
    confidence: Optional[float] = Form(None, ge=0.1, le=1.0)
):
    image = await load_and_preprocess_image(file)
    
    try:
        detection_response = await roboflow_service.detect_ingredients(
            image=image,
            conf_threshold=confidence,
            filter_non_food=True
        )
    except Exception as e:
        logger.error("Lỗi khi gọi Roboflow Serverless Cloud: %s", e)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Lỗi từ dịch vụ Roboflow Serverless Cloud: {str(e)}"
        )
    
    annotated_image = yolo_service.draw_annotations(image, detection_response)
    
    buffer = io.BytesIO()
    annotated_image.save(buffer, format="JPEG", quality=90)
    
    return Response(
        content=buffer.getvalue(),
        media_type="image/jpeg",
        headers={
            "X-Execution-Time-Ms": str(detection_response.execution_time_ms),
            "X-Ingredient-Count": str(detection_response.ingredient_count),
            "X-Model-Used": detection_response.model_name
        }
    )

@router.get(
    "/supported-labels",
    response_model=SupportedLabelsResponse,
    summary="Danh sách các nhãn nguyên liệu được hỗ trợ"
)
async def get_supported_labels():
    labels_dict = yolo_service._labels_map.get("labels", {})
    label_items = [
        LabelInfo(
            name_en=key,
            name_vi=val.get("name_vi", key),
            category=val.get("category", "Khác"),
            is_food=val.get("is_food", True)
        )
        for key, val in labels_dict.items()
    ]
    return SupportedLabelsResponse(
        total_labels=len(label_items),
        labels=label_items
    )
