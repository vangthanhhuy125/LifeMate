# Hướng dẫn Đảm bảo Tính Tái lập (Reproducibility Guide)

Để đảm bảo kết quả chạy lại từ Docker image của Ban tổ chức khớp 100% với kết quả nộp bài:

1. **Cố định Hạt giống Ngẫu nhiên (`src/utils/seed.py`)**:
   - `PYTHONHASHSEED=42`
   - `random.seed(42)`, `np.random.seed(42)`, `torch.manual_seed(42)`, `torch.cuda.manual_seed_all(42)`
   - `torch.backends.cudnn.deterministic = True` và `torch.backends.cudnn.benchmark = False`
2. **Giải mã Tất định (Deterministic Decoding)**:
   - ASR: `beam_size = 1`, `temperature = 0.0`, `do_sample = False`.
   - LLM Planner: `temperature = 0.0`, `do_sample = False`, `num_beams = 1`.
3. **Chế độ Offline Tuyệt đối**:
   - Mọi mô hình và chỉ mục RAG (`artifacts/rag_trajectories.json`) được nạp cục bộ với cờ `local_files_only=True`.