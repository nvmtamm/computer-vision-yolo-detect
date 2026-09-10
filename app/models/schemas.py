from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field

class BoundingBoxCoordinates(BaseModel):
    x_min: int = Field(..., description="Tọa độ pixel góc trên bên trái (trục X)")
    y_min: int = Field(..., description="Tọa độ pixel góc trên bên trái (trục Y)")
    x_max: int = Field(..., description="Tọa độ pixel góc dưới bên phải (trục X)")
    y_max: int = Field(..., description="Tọa độ pixel góc dưới bên phải (trục Y)")

class BoundingBoxItem(BaseModel):
    label: str = Field(..., description="Tên nhãn tiếng Anh gốc từ mô hình")
    confidence: float = Field(..., description="Độ tin cậy của phát hiện (0.0 đến 1.0)")
    box: BoundingBoxCoordinates = Field(..., description="Tọa độ khung giới hạn pixel")
    label_vi: Optional[str] = Field(default=None, description="Tên tiếng Việt tùy chọn")
    category: Optional[str] = Field(default=None, description="Phân nhóm thực phẩm")

class PredictionItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    
    class_name: str = Field(..., alias="class", description="Tên class tiếng Anh gốc từ Cloud Roboflow")
    confidence: float = Field(..., description="Độ tin cậy của phát hiện")
    x: float = Field(..., description="Tọa độ tâm X từ Cloud")
    y: float = Field(..., description="Tọa độ tâm Y từ Cloud")
    width: float = Field(..., description="Chiều rộng từ Cloud")
    height: float = Field(..., description="Chiều cao từ Cloud")
    box: BoundingBoxCoordinates = Field(..., description="Tọa độ 4 góc pixel")
    class_id: Optional[int] = Field(default=None, description="ID class từ mô hình")
    detection_id: Optional[str] = Field(default=None, description="ID phát hiện từ Roboflow")

class IngredientSummary(BaseModel):
    name: str = Field(..., description="Tên nguyên liệu tiếng Anh chuẩn từ Cloud")
    count: int = Field(..., description="Số lượng vật thể phát hiện được trong ảnh")
    max_confidence: float = Field(..., description="Độ tin cậy cao nhất của nguyên liệu này")
    name_en: Optional[str] = Field(default=None, description="Tên tiếng Anh (alias)")
    name_vi: Optional[str] = Field(default=None, description="Tên tiếng Việt tùy chọn")
    category: Optional[str] = Field(default=None, description="Phân nhóm thực phẩm tùy chọn")

class DetectionResponse(BaseModel):
    success: bool = True
    execution_time_ms: float = Field(..., description="Thời gian suy luận mô hình tính bằng mili-giây")
    model_name: str = Field(..., description="Tên hoặc endpoint mô hình AI đang dùng")
    ingredient_count: int = Field(..., description="Tổng số loại nguyên liệu được bóc tách")
    detected_ingredients: List[IngredientSummary] = Field(
        ..., description="Danh sách các nguyên liệu tóm tắt tiếng Anh (dùng cho AI Chef)"
    )
    predictions: List[PredictionItem] = Field(
        default=[], description="Danh sách đối tượng giữ nguyên format Cloud Roboflow"
    )
    bounding_boxes: List[BoundingBoxItem] = Field(
        default=[], description="Chi tiết tọa độ từng vật thể (dùng vẽ khung trên Frontend)"
    )
    fallback_used: bool = Field(default=False, description="Cờ đánh dấu liệu có phải dùng fallback mô hình local không")

class HealthResponse(BaseModel):
    status: str = "ok"
    app_name: str
    version: str
    active_mode: str = "roboflow"
    roboflow_model: Optional[str] = None
    model_loaded: bool
    model_path: str

class LabelInfo(BaseModel):
    name_en: str
    name_vi: str
    category: str
    is_food: bool

class SupportedLabelsResponse(BaseModel):
    total_labels: int
    labels: List[LabelInfo]
