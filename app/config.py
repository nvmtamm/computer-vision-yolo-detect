import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="allow")
    
    APP_NAME: str = "VeggieLife CV Service"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Inference Mode: 'roboflow' (Cloud API) or 'local' (Local YOLOv8)
    INFERENCE_MODE: str = "roboflow"
    
    # Roboflow Cloud API Configuration
    ROBOFLOW_API_KEY: str = "iixXR17kFp48yc3y28np"
    ROBOFLOW_PROJECT_ID: str = "smart-fridge-co7ul-v7nkj"
    ROBOFLOW_MODEL_ID: str = "smart-fridge-co7ul-v7nkj/1"
    ROBOFLOW_API_URL: str = "https://serverless.roboflow.com"
    ROBOFLOW_TIMEOUT_SECONDS: float = 10.0
    
    # Model Weights Path (Falls back to yolov8n.pt if veggie_best.pt not present)
    CUSTOM_WEIGHTS_PATH: Path = BASE_DIR / "weights" / "veggie_best.pt"
    BASELINE_WEIGHTS_PATH: Path = BASE_DIR / "weights" / "yolov8n.pt"
    
    # Inference parameters
    DEFAULT_CONF_THRESHOLD: float = 0.40
    DEFAULT_IOU_THRESHOLD: float = 0.45
    IMAGE_SIZE: int = 640
    
    # Labels metadata
    LABELS_MAP_PATH: Path = BASE_DIR / "data" / "labels_map.json"
    
    # Allowed CORS origins (Frontend and Backend)
    CORS_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5000",
        "http://localhost:5200",
        "https://*.run.app",
        "*"
    ]
    
    @property
    def effective_model_path(self) -> str:
        """Use fine-tuned weights if available, otherwise baseline pretrained."""
        if self.CUSTOM_WEIGHTS_PATH.exists():
            return str(self.CUSTOM_WEIGHTS_PATH)
        if self.BASELINE_WEIGHTS_PATH.exists():
            return str(self.BASELINE_WEIGHTS_PATH)
        # If neither exists locally, use 'yolov8n.pt' which ultralytics can download automatically
        return "yolov8n.pt"

settings = Settings()
