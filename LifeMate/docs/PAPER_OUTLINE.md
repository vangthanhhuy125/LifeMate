# Đề cương Bài báo Kỹ thuật (Technical Paper Outline)

## Tiêu đề
**LifeMate: A Resource-Constrained State-Grounded POMDP Voice Agent with Validate-Before-Call Self-Correction for Spoken Vietnamese Tool Use**

## Cấu trúc Nội dung
1. **Abstract**: Tóm tắt bài toán điều khiển ứng dụng bằng giọng nói tiếng Việt dưới ràng buộc $<6\text{B}$ tham số và cơ chế chấm điểm nghiêm ngặt theo `state_hash`.
2. **Introduction & Problem Formulation**: Định nghĩa bài toán dưới dạng POMDP $(\mathcal{S}, \mathcal{A}, \mathcal{O}, \mathcal{T}, \mathcal{R})$.
3. **Proposed Architecture**:
   - 3.1 Acoustic Code-Switching Normalization & Temporal Grounding.
   - 3.2 Dynamic Schema Pruning for Sub-5B Language Models.
   - 3.3 Hybrid BM25 and App-Overlap Trajectory Retrieval.
   - 3.4 Validate-Before-Call Closed-Loop Execution Guard.
4. **Experiments & Ablation Studies**: Đánh giá độ chính xác State Match, Tool-Call F1 và tốc độ suy luận trên các cấu hình trong `configs/ablations/`.
5. **Conclusion**: Tổng kết hiệu quả của kiến trúc bám sát trạng thái môi trường trên thiết bị biên.