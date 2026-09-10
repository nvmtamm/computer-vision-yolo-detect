import base64
import io
import json
import logging
import time
from typing import Dict, List, Optional, Tuple

import httpx
from PIL import Image

from app.config import settings
from app.models.schemas import (
    BoundingBoxCoordinates,
    BoundingBoxItem,
    DetectionResponse,
    IngredientSummary,
    PredictionItem,
)

logger = logging.getLogger("veggielife.roboflow")

class RoboflowService:
    _instance: Optional["RoboflowService"] = None
    _labels_map: Dict = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RoboflowService, cls).__new__(cls)
            cls._instance._load_labels_map()
        return cls._instance

    def _load_labels_map(self) -> None:
        """Load the bilingual Vietnamese/English label mapping."""
        map_path = settings.LABELS_MAP_PATH
        if map_path.exists():
            try:
                with open(map_path, "r", encoding="utf-8") as f:
                    self._labels_map = json.load(f)
            except Exception as e:
                logger.error("Lỗi khi nạp labels_map.json trong RoboflowService: %s", e)
                self._labels_map = {}

    def _get_label_meta(self, label_en: str) -> Tuple[str, str, bool]:
        """Lookup Vietnamese name, category, and food validity for a label."""
        label_key = label_en.lower().strip()
        labels_dict = self._labels_map.get("labels", {})
        if label_key in labels_dict:
            meta = labels_dict[label_key]
            return meta.get("name_vi", label_en), meta.get("category", "Thực phẩm"), meta.get("is_food", True)
        
        default_meta = self._labels_map.get("default_unknown", {})
        return (
            default_meta.get("name_vi", label_en.capitalize()),
            default_meta.get("category", "Nguyên liệu khác"),
            True
        )

    async def detect_ingredients(
        self,
        image: Image.Image,
        conf_threshold: Optional[float] = None,
        filter_non_food: bool = True
    ) -> DetectionResponse:
        """
        Call Roboflow Serverless Cloud API and transform into standardized VeggieLife schema.
        """
        conf = conf_threshold if conf_threshold is not None else settings.DEFAULT_CONF_THRESHOLD
        
        # 1. Encode PIL Image to JPEG Base64
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        # 2. Build URL & headers for Roboflow Serverless API
        url = (
            f"{settings.ROBOFLOW_API_URL.rstrip('/')}/{settings.ROBOFLOW_MODEL_ID}"
            f"?api_key={settings.ROBOFLOW_API_KEY}&confidence={int(conf * 100)}"
        )
        
        start_time = time.time()
        logger.info("Đang gửi ảnh tới Roboflow Cloud API (Model: %s)...", settings.ROBOFLOW_MODEL_ID)
        
        async with httpx.AsyncClient(timeout=settings.ROBOFLOW_TIMEOUT_SECONDS) as client:
            response = await client.post(
                url,
                content=img_b64,
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )
            
        if response.status_code != 200:
            logger.error("Roboflow API trả về mã lỗi %d: %s", response.status_code, response.text)
            raise RuntimeError(f"Lỗi gọi Roboflow API (Status {response.status_code}): {response.text}")
            
        raw_data = response.json()
        exec_time = (time.time() - start_time) * 1000.0
        
        # 3. Transform predictions preserving Cloud format & standard coordinates
        predictions = raw_data.get("predictions", [])
        cloud_predictions: List[PredictionItem] = []
        bounding_boxes: List[BoundingBoxItem] = []
        ingredients_agg: Dict[str, Dict] = {}
        
        for p in predictions:
            confidence = round(float(p.get("confidence", 0.0)), 4)
            if confidence < conf:
                continue
                
            # Keep original raw class from Roboflow (e.g. 'Apple', 'Milk', 'Eggs')
            label_en = str(p.get("class", "unknown")).strip()
            label_vi, category, is_food = self._get_label_meta(label_en)
            
            if filter_non_food and not is_food:
                continue
                
            # Convert center-based (x, y, w, h) to corner coordinates (x_min, y_min, x_max, y_max)
            cx, cy = float(p.get("x", 0)), float(p.get("y", 0))
            w, h = float(p.get("width", 0)), float(p.get("height", 0))
            
            x_min = max(0, int(cx - w / 2.0))
            y_min = max(0, int(cy - h / 2.0))
            x_max = min(image.width, int(cx + w / 2.0))
            y_max = min(image.height, int(cy + h / 2.0))
            
            box_coord = BoundingBoxCoordinates(
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max
            )
            
            # Preserve raw Cloud prediction item
            pred_item = PredictionItem(
                class_name=label_en,
                confidence=confidence,
                x=cx,
                y=cy,
                width=w,
                height=h,
                box=box_coord,
                class_id=p.get("class_id"),
                detection_id=p.get("detection_id")
            )
            cloud_predictions.append(pred_item)
            
            # Bounding box item for visual drawing
            bbox_item = BoundingBoxItem(
                label=label_en,
                confidence=confidence,
                box=box_coord,
                label_vi=label_vi,
                category=category
            )
            bounding_boxes.append(bbox_item)
            
            # Aggregate counts for AI prompt using English name
            norm_key = label_en.lower()
            if norm_key not in ingredients_agg:
                ingredients_agg[norm_key] = {
                    "name": label_en,
                    "name_en": label_en,
                    "name_vi": label_vi,
                    "category": category,
                    "count": 1,
                    "max_confidence": confidence
                }
            else:
                ingredients_agg[norm_key]["count"] += 1
                if confidence > ingredients_agg[norm_key]["max_confidence"]:
                    ingredients_agg[norm_key]["max_confidence"] = confidence
                    
        ingredient_summaries = [
            IngredientSummary(**val) for val in ingredients_agg.values()
        ]
        
        return DetectionResponse(
            success=True,
            execution_time_ms=round(exec_time, 2),
            model_name=f"roboflow:{settings.ROBOFLOW_MODEL_ID}",
            ingredient_count=len(ingredient_summaries),
            detected_ingredients=ingredient_summaries,
            predictions=cloud_predictions,
            bounding_boxes=bounding_boxes
        )

roboflow_service = RoboflowService()
