import sys
import json
import yaml
import argparse
from pathlib import Path
from tqdm import tqdm

from src.utils.reproducibility import lock_deterministic_environment
from src.pipeline.lifemate_pomdp_pipeline import LifeMatePOMDPPipeline

def main():
    parser = argparse.ArgumentParser(description="Run 3-Stage POMDP Voice Agent Pipeline")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml")
    parser.add_argument("--data_root", type=str, required=True, help="Thư mục gốc chứa các split (train/public_test/private_test)")
    parser.add_argument("--split", type=str, default="public_test")
    parser.add_argument("--output", type=str, default="result/submission.jsonl")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    lock_deterministic_environment(config.get("experiment", {}).get("seed", 42))

    root_path = Path(args.data_root)
    split_dir = root_path / args.split
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Nạp zaic_tools từ thư mục của chính split đó theo đúng quy định của BTC
    sys.path.insert(0, str(split_dir))
    from zaic_tools import ToolEnv

    catalog_file = split_dir / f"{args.split}_tools_catalog.json"
    with open(catalog_file, "r", encoding="utf-8") as f:
        full_catalog = json.load(f)

    pipeline = LifeMatePOMDPPipeline(config)
    input_jsonl = split_dir / f"{args.split}.jsonl"

    with open(input_jsonl, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc=f"Inference [{args.split}]"):
            if not line.strip():
                continue
            raw_sample = json.loads(line)
            # Khởi tạo đúng 1 ToolEnv duy nhất cho mỗi mẫu
            env = ToolEnv(args.split, raw_sample["id"], root=str(root_path))

            sub_record, _ = pipeline.infer_single_sample(
                raw_sample=raw_sample,
                env=env,
                split_dir=split_dir,
                full_catalog=full_catalog
            )
            fout.write(json.dumps(sub_record, ensure_ascii=False) + "\n")
            fout.flush()

if __name__ == "__main__":
    main()
