# Thư mục chứa trọng số mô hình YOLO (Model Weights)

## Các file trọng số:
1. `yolov8n.pt`: 
   - Mô hình YOLOv8 Nano Pretrained (COCO Dataset ~ 6.2MB).
   - Dùng để chạy baseline tức thì, kiểm thử API và luồng tích hợp với ASP.NET Core & React.
   - Sẽ được tải tự động nếu chưa có sẵn.

2. `veggie_best.pt`:
   - Trọng số sau khi Fine-tune tập dữ liệu thực phẩm & rau củ chay Việt Nam (Đậu hũ, nấm rơm, mướp, bí,...).
   - Khi file `veggie_best.pt` xuất hiện trong thư mục này, hệ thống sẽ tự động ưu tiên nạp mô hình này thay thế cho `yolov8n.pt`.
