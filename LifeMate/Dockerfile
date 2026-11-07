FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace
COPY . /workspace

# Cài đặt gói nội bộ ở chế độ offline-ready
RUN pip install --no-cache-dir -e .

# Chạy inference hoàn toàn offline khi BTC mount dữ liệu vào /data
CMD ["python", "scripts/run_inference.py", "--config", "configs/default_config.yaml", "--data_root", "/data", "--split", "private_test", "--output", "result/submission.jsonl"]
