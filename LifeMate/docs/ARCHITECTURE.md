```markdown
# Kiến trúc Hệ thống LifeMate POMDP Voice Agent

## 1. Mô hình Hóa Toán học (POMDP Formulation)

Tại mỗi mẫu dữ liệu, hệ thống tương tác với môi trường ẩn $\mathcal{E}$ (`zaic_tools.ToolEnv`) qua bộ tuple $(\mathcal{S}, \mathcal{A}, \mathcal{O}, \mathcal{T}, \mathcal{R})$:
- **Trạng thái niềm tin (Belief State $b_k$)**: Lưu trữ khẩu lệnh đã chuẩn hóa NFC, các thực thể trích xuất (`extracted_clues`), tập các thực thể `x-ref` đã thu thập từ môi trường (`known_entities`), và lịch sử các bước thực thi $H_k = \{(a_1, o_1), \dots, (a_k, o_k)\}$.
- **Không gian hành động ($\mathcal{A}$)**:
  1. `CALL_TOOL(tool_name, arguments)`: Gọi công cụ `read` để tăng thông tin (Information Gain) hoặc công cụ `write` để thay đổi trạng thái ứng dụng.
  2. `ASK_MISSING("sys.ask_missing_info", {"tool_name": ..., "params": [...]})`: Hỏi lại người dùng khi thiếu tham số bắt buộc không thể tra cứu.
  3. `TERMINATE`: Kết thúc chuỗi hành động (hoặc trả về `[]` khi yêu cầu nằm ngoài phạm vi `installed_apps`).

## 2. Ba Giai đoạn Xử lý (3-Stage Pipeline)

1. **Stage 1 (`stage1_perception_and_grounding.py`)**:
   - `AcousticFrontend`: Giải mã WAV 16kHz sang văn bản tiếng Việt và chuẩn hóa từ điển pha trộn ngôn ngữ Anh-Việt (`code_switching_vi_en.yaml`).
   - `NLUNormalizer`: Chuẩn hóa Unicode NFC, quy đổi số đếm/tiền tệ và giải quyết biểu thức thời gian tương đối sang chuẩn ISO-8601.
   - `DynamicSchemaPruner`: Cắt tỉa `<split>_tools_catalog.json` theo `context.installed_apps`.
   - `AppOverlapRetriever`: Truy xuất các quỹ đạo mẫu tương đồng làm ngữ cảnh Few-Shot.
2. **Stage 2 (`stage2_pomdp_planner.py`)**:
   - Kết hợp mô hình ngôn ngữ nhỏ ($<5\text{B}$) và bộ ước lượng Expected Utility để chọn hành động tiếp theo, đảm bảo quy tắc **Read-Before-Write** đối với mọi tham số có thuộc tính `x-ref`.
3. **Stage 3 (`stage3_closed_loop_executor.py`)**:
   - Kiểm tra hành động qua `ValidatorGuard` (`env.validate()`) trước khi thực thi `env.call()`.
   - Trích xuất kết quả cuối cùng từ `env.get_tool_calls()` và `env.get_state_hash()`.