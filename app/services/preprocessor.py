import io
from PIL import Image, ImageOps
from fastapi import HTTPException, UploadFile, status

MAX_FILE_SIZE_MB = 15
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
    "application/octet-stream"
}

def validate_image_file(file: UploadFile) -> None:
    """Validate file extension and content type."""
    if file.filename:
        ext = "." + file.filename.split(".")[-1].lower() if "." in file.filename else ""
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Định dạng file '{ext}' không được hỗ trợ. Vui lòng tải ảnh JPG, PNG hoặc WEBP."
            )

async def load_and_preprocess_image(file: UploadFile) -> Image.Image:
    """
    Read bytes from UploadFile, validate size, auto-correct EXIF orientation,
    and convert to RGB PIL Image.
    """
    validate_image_file(file)
    
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Dung lượng ảnh vượt quá giới hạn cho phép ({MAX_FILE_SIZE_MB}MB)."
        )
    
    try:
        image = Image.open(io.BytesIO(contents))
        # Handle smartphone EXIF orientation tag automatically
        image = ImageOps.exif_transpose(image)
        # Ensure RGB format (handles PNG RGBA or grayscale images)
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Không thể đọc nội dung ảnh: {str(e)}"
        )
