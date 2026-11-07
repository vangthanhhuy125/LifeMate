# Phân loại Lỗi và Cơ chế Phòng ngừa trong LifeMate

1. **Lỗi Ảo giác ID Tham chiếu (`X-Ref Hallucination`)**:
   - *Nguyên nhân*: LLM tự sinh ID thiết bị/danh bạ (ví dụ `ac_01`) khi chưa gọi công cụ `read`.
   - *Khắc phục*: `ValidatorGuard.check_xref_grounding()` chặn mọi lệnh `write` có tham số `x-ref` nếu ID chưa xuất hiện trong `BeliefState.known_entities`, đồng thời tự động kích hoạt công cụ `read` tương ứng.
2. **Lỗi Sai Kiểu dữ liệu / Thiếu Tham số Bắt buộc (`Schema Validation Failure`)**:
   - *Nguyên nhân*: Truyền chuỗi thay vì số nguyên hoặc thiếu trường `required` / `x-one-of`.
   - *Khắc phục*: Tiền kiểm tra bằng `env.validate(tool_name, arguments)` trước khi gọi `env.call()`.
3. **Lỗi Lệch Chuẩn Chuỗi Tiếng Việt & Thời gian (`Encoding & Temporal Drift`)**:
   - *Nguyên nhân*: Tổ hợp dấu Unicode NFD hoặc sai lệch múi giờ ISO-8601 làm sai `state_hash`.
   - *Khắc phục*: Chuẩn hóa toàn bộ chuỗi đầu vào/đầu ra về Unicode NFC và tính toán thời gian ISO-8601 trực tiếp từ `context.current_time`.