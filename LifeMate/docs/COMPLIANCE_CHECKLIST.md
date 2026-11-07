# Danh mục Rà soát Tuân thủ Quy định Cuộc thi (Compliance Checklist)

| Hạng mục Quy định | Giới hạn Đề bài | Thiết kế trong LifeMate | Trạng thái |
| :--- | :--- | :--- | :--- |
| Tổng kích thước mô hình | $< 6.0\text{B}$ tham số | ASR ($0.24\text{B}$) + LLM ($3.09\text{B}$) = $3.33\text{B}$ | Đạt |
| Kích thước mô hình LLM | $< 5.0\text{B}$ tham số | Qwen2.5-3B-Instruct ($3.09\text{B}$) | Đạt |
| Độ dài ngữ cảnh (Context Length) | $\le 16.384$ tokens | Kiểm soát cứng tại `TokenBudgetManager` ($\le 12.288$ safe tokens) | Đạt |
| Truy cập dữ liệu môi trường | Cấm đọc trực tiếp `state.json` | Chỉ gọi qua API của `zaic_tools.ToolEnv` | Đạt |
| Số lượng `ToolEnv` mỗi mẫu | Đúng 01 instance/mẫu | Khởi tạo 01 lần duy nhất tại `run_inference.py` | Đạt |
| Thời gian xử lý mỗi mẫu | $\le 30$ giây / mẫu | Ngắt an toàn ở ngưỡng $28.0$ giây tại `Stage3ClosedLoopExecutor` | Đạt |
| Chế độ thực thi Docker | Offline 100%, không dùng mạng | Đặt biến môi trường `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` | Đạt |
| Tính tái lập (Reproducibility) | Cố định seed, Greedy Decoding | `seed=42`, `temperature=0.0`, `do_sample=False` | Đạt |