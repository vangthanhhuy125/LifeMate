# LifeMate: Resource-Constrained State-Grounded POMDP Voice Agent (Zalo AI Challenge 2026)

Hệ thống AI Agent điều khiển thiết bị bằng giọng nói tiếng Việt theo mô hình **Quá trình Ra quyết định Markov Quan sát Một phần (POMDP)**, được thiết kế chuyên biệt cho bài toán **LifeMate – Zalo AI Challenge 2026**.

## 1. Điểm nhấn Kiến trúc

1. **Validate-Before-Call Self-Correction (`ValidatorGuard`)**: Mọi hành động dự kiến đều được kiểm tra qua `env.validate(tool_name, arguments)` kết hợp kiểm tra ràng buộc `required`, `x-one-of` và sự tồn tại thực tế của `x-ref` ID trong `BeliefState` trước khi gọi `env.call()`. Cơ chế này triệt tiêu các lệnh gọi lỗi schema làm tăng số lệnh sai $U$ (bảo toàn Precision $P = \frac{M}{M+U}$).
2. **Dynamic Schema Pruning (`DynamicSchemaPruner`)**: Tự động lọc danh mục công cụ theo `context.installed_apps` và giữ lại các công cụ `read` cung cấp `x-ref` cùng `sys.ask_missing_info`, giúp giảm hơn 70% lượng prompt tokens và nằm gọn dưới giới hạn 16K context.
3. **Hybrid BM25 + App-Overlap Few-Shot RAG (`AppOverlapRetriever`)**: Truy xuất các quỹ đạo mẫu từ tập huấn luyện có sự tương đồng về ngữ nghĩa khẩu lệnh và giao thoa tập ứng dụng cài đặt.
4. **Tuân thủ tuyệt đối quy định `zaic_tools`**:
   - Khởi tạo đúng **01 đối tượng `ToolEnv` duy nhất** cho mỗi mẫu dữ liệu.
   - **Tuyệt đối không** đọc, sao chép hay phân tích trực tiếp tệp `<split>_environment/<id>/state.json`.
   - Tổng tham số toàn hệ thống $< 6\text{B}$ (PhoWhisper-small $0.24\text{B}$ + Qwen2.5-3B-Instruct $3.09\text{B}$ = $3.33\text{B}$), giải mã tất định (`temperature = 0`), chạy hoàn toàn offline trong Docker container dưới 30 giây/mẫu.

## 2. Cấu trúc Thư mục (Không chứa thư mục `data/` cứng)
LifeMate/
├── pyproject.toml
├── Makefile
├── Dockerfile
├── README.md
├── configs/
│   ├── default_config.yaml
│   ├── lexicon/
│   │   └── code_switching_vi_en.yaml
│   ├── prompts/
│   │   ├── policy_system_prompt.txt
│   │   └── pomdp_policy_prompt.txt
│   └── ablations/
│       ├── ablation_no_schema_pruning.yaml
│       ├── ablation_no_validate_guard.yaml
│       ├── no_constrained_decoding.yaml
│       ├── no_pruning.yaml
│       ├── no_rag.yaml
│       ├── no_time_normalizer.yaml
│       ├── no_validate.yaml
│       ├── no_xref_guard.yaml
│       ├── oracle_asr.yaml
│       └── zero_shot_llm.yaml
├── src/
│   ├── __init__.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── seed.py
│   │   ├── reproducibility.py
│   │   ├── pomdp_types.py
│   │   └── token_budget.py
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── acoustic_frontend.py
│   │   ├── nlu_normalizers.py
│   │   ├── context_memory.py
│   │   ├── schema_pruner.py
│   │   ├── dynamic_schema_pruner.py
│   │   ├── retriever_bm25.py
│   │   ├── app_overlap_retriever.py
│   │   └── validator_guard.py
│   ├── stages/
│   │   ├── __init__.py
│   │   ├── stage1_perception_and_grounding.py
│   │   ├── stage2_pomdp_planner.py
│   │   └── stage3_closed_loop_executor.py
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── pipeline.py
│   │   └── lifemate_pomdp_pipeline.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── pomdp_trajectory_builder.py
│   │   └── train_lora_sft.py
│   └── evaluation/
│       ├── __init__.py
│       └── official_scorer.py
├── scripts/
│   ├── 00_inspect_data.py
│   ├── prepare_rag_and_sft.py
│   ├── run_inference.py
│   └── evaluate_and_ablate.py
├── tests/
│   ├── __init__.py
│   ├── fakes/
│   │   ├── __init__.py
│   │   └── fake_env.py
│   ├── test_pomdp.py
│   └── test_pomdp_pipeline.py
└── docs/
    ├── ARCHITECTURE.md
    ├── COMPLIANCE_CHECKLIST.md
    ├── ERROR_TAXONOMY.md
    ├── PAPER_OUTLINE.md
    └── REPRODUCIBILITY.md


## 3. Cài đặt và Kiểm thử

```bash
# 1. Cài đặt gói và thư viện phụ trợ
make install

# 2. Chạy toàn bộ bộ kiểm thử đơn vị và tích hợp (Unit & Pipeline Tests)
make test

