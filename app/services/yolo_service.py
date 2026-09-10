import io
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from app.config import settings
from app.models.schemas import (
    BoundingBoxCoordinates,
    BoundingBoxItem,
    DetectionResponse,
    IngredientSummary,
)

logger = logging.getLogger("veggielife.cv")

class YOLOService:
    _instance: Optional["YOLOService"] = None
    _model: Optional[YOLO] = None
    _labels_map: Dict = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(YOLOService, cls).__new__(cls)
            cls._instance._load_labels_map()
        return cls._instance

    def _load_labels_map(self) -> None:
        """Load the bilingual Vietnamese/English label mapping."""
        map_path = settings.LABELS_MAP_PATH
        if map_path.exists():
            try:
                with open(map_path, "r", encoding="utf-8") as f:
                    self._labels_map = json.load(f)
                logger.info("Đã tải thành công bản đồ nhãn: %s", map_path)
            except Exception as e:
                logger.error("Lỗi khi đọc file labels_map.json: %s", e)
                self._labels_map = {}
        else:
            logger.warning("Không tìm thấy file labels_map.json tại %s", map_path)
            self._labels_map = {}

    def get_model(self) -> YOLO:
        """Lazy load YOLO model or return cached instance."""
        if self._model is None:
            model_path = settings.effective_model_path
            logger.info("Đang nạp mô hình YOLO từ: %s", model_path)
            start = time.time()
            self._model = YOLO(model_path)
            logger.info("Nạp mô hình hoàn tất trong %.2f giây", time.time() - start)
        return self._model

    def warmup(self) -> None:
        """Warm up the model with a blank image to avoid 1st request cold delay."""
        try:
            model = self.get_model()
            dummy_img = Image.new("RGB", (320, 320), color=(128, 128, 128))
            model.predict(source=dummy_img, imgsz=320, verbose=False)
            logger.info("Khởi động (warmup) mô hình YOLO thành công!")
        except Exception as e:
            logger.warning("Không thể warmup mô hình: %s", e)

    def _get_label_meta(self, label_en: str) -> Tuple[str, str, bool]:
        """Lookup Vietnamese name, category, and food validity for a label."""
        labels_dict = self._labels_map.get("labels", {})
        if label_en in labels_dict:
            meta = labels_dict[label_en]
            return meta.get("name_vi", label_en), meta.get("category", "Thực phẩm"), meta.get("is_food", True)
        
        # Fallback for unknown items
        default_meta = self._labels_map.get("default_unknown", {})
        return (
            default_meta.get("name_vi", label_en.capitalize()),
            default_meta.get("category", "Nguyên liệu khác"),
            True
        )

    def detect_ingredients(
        self,
        image: Image.Image,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        filter_non_food: bool = True
    ) -> DetectionResponse:
        """
        Run inference on image and return structured ingredients and bounding boxes.
        """
        conf = conf_threshold if conf_threshold is not None else settings.DEFAULT_CONF_THRESHOLD
        iou = iou_threshold if iou_threshold is not None else settings.DEFAULT_IOU_THRESHOLD
        
        model = self.get_model()
        start_time = time.time()
        
        # Run inference
        results = model.predict(
            source=image,
            conf=conf,
            iou=iou,
            imgsz=settings.IMAGE_SIZE,
            verbose=False
        )
        
        exec_time = (time.time() - start_time) * 1000.0
        
        bounding_boxes: List[BoundingBoxItem] = []
        ingredients_agg: Dict[str, Dict] = {}
        
        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes
            names = result.names
            
            for box in boxes:
                cls_id = int(box.cls[0].item())
                confidence = round(float(box.conf[0].item()), 4)
                label_en = names.get(cls_id, f"class_{cls_id}").lower().strip()
                
                label_vi, category, is_food = self._get_label_meta(label_en)
                
                # If configured, skip non-food objects like bottles, cups, chairs, etc.
                if filter_non_food and not is_food:
                    continue
                
                coords = box.xyxy[0].tolist()
                x_min, y_min, x_max, y_max = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
                w, h = float(x_max - x_min), float(y_max - y_min)
                cx, cy = float(x_min + w / 2.0), float(y_min + h / 2.0)
                
                bbox_coord = BoundingBoxCoordinates(
                    x_min=x_min,
                    y_min=y_min,
                    x_max=x_max,
                    y_max=y_max
                )
                
                # Keep English label from model
                capitalized_label = label_en.capitalize()
                bbox_item = BoundingBoxItem(
                    label=capitalized_label,
                    confidence=confidence,
                    box=bbox_coord,
                    label_vi=label_vi,
                    category=category
                )
                bounding_boxes.append(bbox_item)
                
                # Aggregate counts & max confidence for Gemini AI prompt
                if label_en not in ingredients_agg:
                    ingredients_agg[label_en] = {
                        "name": capitalized_label,
                        "name_en": label_en,
                        "name_vi": label_vi,
                        "category": category,
                        "count": 1,
                        "max_confidence": confidence
                    }
                else:
                    ingredients_agg[label_en]["count"] += 1
                    if confidence > ingredients_agg[label_en]["max_confidence"]:
                        ingredients_agg[label_en]["max_confidence"] = confidence

        ingredient_summaries = [
            IngredientSummary(**val) for val in ingredients_agg.values()
        ]
        
        return DetectionResponse(
            success=True,
            execution_time_ms=round(exec_time, 2),
            model_name=settings.effective_model_path,
            ingredient_count=len(ingredient_summaries),
            detected_ingredients=ingredient_summaries,
            bounding_boxes=bounding_boxes
        )

    def draw_annotations(
        self,
        image: Image.Image,
        detection_response: DetectionResponse
    ) -> Image.Image:
        """
        Draw color-coded bounding boxes and labels on top of the image with dynamic
        line thickness and scalable font size proportional to image resolution.
        Labels are displayed in pure English as returned by Cloud/YOLO.
        """
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        
        # Calculate dynamic line thickness and font size based on image resolution
        min_dim = min(image.width, image.height)
        line_thickness = max(4, int(min_dim / 250))
        font_size = max(20, int(min_dim / 45))
        
        # Attempt to load a clean TrueType font, falling back to default scalable font
        font = None
        candidate_fonts = [
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ]
        for c in candidate_fonts:
            if Path(c).exists():
                try:
                    font = ImageFont.truetype(c, font_size)
                    break
                except Exception:
                    pass
                    
        if font is None:
            try:
                font = ImageFont.load_default(size=font_size)
            except TypeError:
                font = ImageFont.load_default()

        # High contrast palette for bounding boxes
        palette = [
            (231, 76, 60),   # Bright Red
            (46, 204, 113),  # Emerald Green
            (52, 152, 219),  # Bright Blue
            (241, 196, 15),  # Sunflower Yellow
            (155, 89, 182),  # Amethyst Purple
            (230, 126, 34),  # Orange
            (26, 188, 156),  # Turquoise
        ]
        
        for idx, item in enumerate(detection_response.bounding_boxes):
            box = item.box
            color = palette[idx % len(palette)]
            
            # 1. Draw thick bounding box
            draw.rectangle(
                [(box.x_min, box.y_min), (box.x_max, box.y_max)],
                outline=color,
                width=line_thickness
            )
            
            # 2. Text in pure English with confidence percentage
            text = f"{item.label} {int(item.confidence * 100)}%"
            
            # Compute badge dimensions with padding
            padding_x = max(6, int(font_size * 0.25))
            padding_y = max(4, int(font_size * 0.15))
            
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
            
            badge_h = text_h + 2 * padding_y
            badge_w = text_w + 2 * padding_x
            
            # Position badge above box, or inside if too close to top edge
            badge_y0 = box.y_min - badge_h if box.y_min >= badge_h else box.y_min
            badge_y1 = badge_y0 + badge_h
            badge_x0 = max(0, box.x_min)
            badge_x1 = min(image.width, badge_x0 + badge_w)
            
            # Draw badge background
            draw.rectangle([badge_x0, badge_y0, badge_x1, badge_y1], fill=color)
            
            # Draw crisp white text inside badge
            draw.text(
                (badge_x0 + padding_x, badge_y0 + padding_y),
                text,
                font=font,
                fill=(255, 255, 255)
            )
            
        # 3. Draw Watermark Badge at Top-Left to explicitly identify Model Source
        model_name = detection_response.model_name.replace("roboflow:", "")
        watermark_text = f" MODEL: ROBOFLOW CLOUD ({model_name}) "
        tag_font_size = max(18, int(font_size * 0.75))
        tag_font = None
        for c in candidate_fonts:
            if Path(c).exists():
                try:
                    tag_font = ImageFont.truetype(c, tag_font_size)
                    break
                except Exception:
                    pass
        if tag_font is None:
            try:
                tag_font = ImageFont.load_default(size=tag_font_size)
            except TypeError:
                tag_font = ImageFont.load_default()

        tag_bbox = draw.textbbox((0, 0), watermark_text, font=tag_font)
        tag_w = tag_bbox[2] - tag_bbox[0]
        tag_h = tag_bbox[3] - tag_bbox[1]
        
        margin = 15
        pad_x = 12
        pad_y = 8
        draw.rectangle(
            [margin, margin, margin + tag_w + 2 * pad_x, margin + tag_h + 2 * pad_y],
            fill=(15, 23, 42),
            outline=(56, 189, 248),
            width=3
        )
        draw.text(
            (margin + pad_x, margin + pad_y),
            watermark_text,
            font=tag_font,
            fill=(255, 255, 255)
        )
        
        return annotated

yolo_service = YOLOService()
