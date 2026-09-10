# Hướng Dẫn Tích Hợp Gemini 2.5 Flash API Sau Khi Có Kết Quả YOLO (Flow 4 - F-07)

Sau khi dịch vụ Computer Vision YOLOv8 trả về danh sách nguyên liệu:
`["Đậu phụ (Đậu hũ)", "Cà chua", "Nấm", "Bông cải xanh"]`

Backend ASP.NET Core 8 sẽ kết hợp thông tin trên với hồ sơ dinh dưỡng của người dùng (`UserProfile`: dị ứng, trường phái ăn chay, mục tiêu calo) để gửi prompt sang **Google Gemini 2.5 Flash API**.

---

## 1. Mẫu System Instruction & Prompt gửi Gemini 2.5 Flash

### System Instruction:
```text
Bạn là Bếp trưởng AI của VeggieLife ("Empty the Fridge" AI Chef). Nhiệm vụ của bạn là sáng tạo các món ăn chay thơm ngon, cân bằng dinh dưỡng và dễ nấu, dựa trên danh sách nguyên liệu còn sót lại trong tủ lạnh mà người dùng chụp được. 

Nguyên tắc bắt buộc:
1. Món ăn phải 100% tuân thủ trường phái ăn chay của người dùng (Thuần chay / Có trứng sữa / Kiêng ngũ vị tân).
2. Tuyệt đối không sử dụng các nguyên liệu thuộc danh sách dị ứng.
3. Ưu tiên tận dụng tối đa các nguyên liệu có sẵn trong tủ lạnh để hạn chế lãng phí thực phẩm. Chỉ gợi ý thêm các gia vị cơ bản thường có trong gian bếp Việt (muối, hạt nêm chay, nước tương, dầu ăn, tiêu).
4. Phải trả về dữ liệu đúng định dạng JSON có cấu trúc để Frontend hiển thị thẻ món ăn đẹp mắt.
```

### Prompt Input (Do Backend lắp ráp tự động):
```json
{
  "user_profile": {
    "dietary_preference": "Vegan (Thuần chay)",
    "strictness": "Kiêng ngũ vị tân (Không hành, hẹ, tỏi, kiệu, nén)",
    "allergies": ["Đậu phộng", "Mè"],
    "target_calories": "450 - 550 kcal / bữa"
  },
  "fridge_ingredients_detected": [
    { "name": "Đậu phụ (Đậu hũ)", "count": 2, "category": "Đạm thực vật" },
    { "name": "Cà chua", "count": 3, "category": "Rau củ" },
    { "name": "Nấm rơm", "count": 1, "category": "Đạm thực vật / Rau củ" },
    { "name": "Bông cải xanh", "count": 1, "category": "Rau củ" }
  ]
}
```

---

## 2. Cấu trúc JSON Output Mong Đợi từ Gemini 2.5 Flash

Thiết lập thuộc tính `response_mime_type: "application/json"` trong Gemini SDK để Gemini trả về JSON chuẩn:

```json
{
  "recipe_name": "Đậu Hũ Sốt Cà Chua Kèm Bông Cải Xanh Hấp",
  "summary": "Món ăn thanh đạm, đậm đà vị chua ngọt tự nhiên từ cà chua kết hợp độ mềm béo của đậu hũ và độ giòn ngọt của súp lơ.",
  "cooking_time_minutes": 25,
  "difficulty": "Dễ",
  "serving_size": "2 người",
  "estimated_calories": 480,
  "nutrition": {
    "protein_g": 24,
    "carbs_g": 35,
    "fat_g": 14,
    "fiber_g": 8
  },
  "ingredients_used": [
    { "name": "Đậu hũ", "amount": "2 miếng", "status": "Có sẵn trong tủ lạnh" },
    { "name": "Cà chua", "amount": "3 quả", "status": "Có sẵn trong tủ lạnh" },
    { "name": "Nấm rơm", "amount": "100g", "status": "Có sẵn trong tủ lạnh" },
    { "name": "Bông cải xanh", "amount": "1 cây nhỏ", "status": "Có sẵn trong tủ lạnh" }
  ],
  "pantry_staples_needed": [
    "Dầu thực vật (2 thìa canh)",
    "Nước tương tamari (1.5 thìa canh)",
    "Hạt nêm nấm chay (1 thìa cà phê)",
    "Đường vàng (1/2 thìa cà phê)"
  ],
  "instructions": [
    { "step": 1, "action": "Sơ chế", "detail": "Đậu hũ cắt miếng vuông vừa ăn. Cà chua băm nhuyễn hoặc thái hạt lựu. Bông cải xanh tách nhánh nhỏ ngâm nước muối loãng." },
    { "step": 2, "action": "Áp chảo đậu hũ", "detail": "Làm nóng chảo với chút dầu, áp chảo nhẹ các mặt đậu hũ cho vàng giòn mép rồi gắp ra đĩa." },
    { "step": 3, "action": "Nấu sốt cà nấm", "detail": "Cho cà chua vào xào nhừ tạo màu, thêm nấm rơm, nêm nước tương, hạt nêm nấm và chút nước xâm xấp. Đun sôi liu riu 5 phút." },
    { "step": 4, "action": "Om đậu & luộc rau", "detail": "Cho đậu hũ vào chảo sốt cà chua, rim nhỏ lửa 5-7 phút cho ngấm gia vị. Trong lúc đó, luộc hoặc hấp chín tới bông cải xanh." },
    { "step": 5, "action": "Trình bày", "detail": "Bày đậu hũ sốt cà chua ra đĩa sâu lòng, xếp bông cải xanh xung quanh và thưởng thức nóng cùng cơm lứt." }
  ],
  "chef_tips": "Vì bạn kiêng ngũ vị tân, vị thơm của món ăn đến từ nấm xào kỹ và vị ngọt tự nhiên của cà chua chín mọng mà hoàn toàn không cần hành tỏi."
}
```
