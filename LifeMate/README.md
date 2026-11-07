# LifeMate State-Grounded POMDP Research Framework (v2.0)

Khung mã nguồn nghiên cứu khoa học hoàn chỉnh cho bài toán **Spoken Vietnamese Task-Oriented Voice Agent**, được thiết kế đồng nhất 100% với **Problem Formulation (POMDP)**, **Input/Output** và **3 Đóng góp học thuật (Contributions)**.

## 1. Cấu trúc Thư mục (Không chứa thư mục `data/` cứng)
```text
LifeMate_POMDP_Research_Framework-v2/
├── configs/
│   ├── default_config.yaml                     # Cấu hình chuẩn kết hợp cả 3 Contributions
│   ├── ablations/                              # Cấu hình chạy Ablation Studies cho bài báo
│   │   ├── ablation_no_validate_guard.yaml     # Bóc tách Contribution 1 (Tắt Validate-Before-Call)
│   │   └── ablation_no_schema_pruning.yaml     # Bóc tách Contribution 2 (Tắt Dynamic Schema Pruning)
│   └── prompts/
│       └── pomdp_policy_prompt.txt             # System Prompt định nghĩa không gian hành động POMDP
├── src/
│   ├── pipeline/
│   │   └── lifemate_pomdp_pipeline.py          # Bộ điều phối Pipeline 3 Stage khép kín
│   ├── stages/                                 # [TRỌNG TÂM] 3 STAGE CỦA PIPELINE
│   │   ├── stage1_perception_and_grounding.py  # Stage 1: ASR + Code-Switching + Schema Pruner S(K) + RAG
│   │   ├── stage2_pomdp_planner.py             # Stage 2: Constrained POMDP Policy Planner (SLM < 5B)
│   │   └── stage3_closed_loop_executor.py      # Stage 3: Validate-Before-Call Self-Correction & ToolEnv
│   ├── modules/                                # Các mô-đun thuật toán thành phần
│   │   ├── acoustic_frontend.py                # PhoWhisper-small (16kHz) + Từ điển chuẩn hóa Anh-Việt
│   │   ├── dynamic_schema_pruner.py            # Cắt tỉa S(K) và phân tích ràng buộc x-ref / x-match
│   │   └── app_overlap_retriever.py            # BM25 RAG tích hợp trọng số giao thoa ứng dụng (App-Overlap)
│   ├── training/
│   │   └── pomdp_trajectory_builder.py         # Xây dựng dữ liệu quỹ đạo đa bước (C, S(K), H_{k-1}, A -> a_k)
│   ├── evaluation/
│   │   └── official_scorer.py                  # Công thức chấm điểm BTC (State Hash, Token-F1 > 0.6, P, R, U)
│   └── utils/
│       ├── pomdp_types.py                      # Định nghĩa Dataclass cho C, S(K), POMDPAction, TransitionStep
│       └── reproducibility.py                  # Cố định toàn bộ Random Seed (Python, NumPy, PyTorch, CUDA)
├── scripts/
│   ├── prepare_rag_and_sft.py                  # Tạo chỉ mục RAG & sinh dữ liệu huấn luyện LoRA SFT
│   ├── run_inference.py                        # Chạy Pipeline 3 Stage trên split bất kỳ để xuất submission.jsonl
│   └── evaluate_and_ablate.py                 # Chấm điểm và xuất bảng phân tích P, R, U, State Match Rate
├── tests/
│   └── test_pomdp_pipeline.py                  # Unit Tests kiểm chứng tự động 3 Contributions & Scorer
├── pyproject.toml                              # Quản lý thư viện chuẩn (hỗ trợ `uv` và `pip`)
└── Dockerfile                                  # Đóng gói container chạy hoàn toàn offline
```
