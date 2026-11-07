"""Supervised Fine-Tuning (SFT) with LoRA for sub-5B POMDP Policy LLMs."""

import json
from pathlib import Path
from typing import Any, Dict, List


class LoRASFTTrainer:
    """Configures and executes parameter-efficient LoRA fine-tuning on POMDP step trajectories."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        train_cfg = config.get("training", {})
        plan_cfg = config.get("pomdp_planner", {})

        self.base_model_path = str(
            plan_cfg.get("llm_model_name_or_path", "models/Qwen2.5-3B-Instruct")
        )
        self.output_dir = str(plan_cfg.get("lora_adapter_path", "artifacts/lora_adapter"))
        self.sft_dataset_path = str(
            train_cfg.get("sft_output_path", "artifacts/sft_pomdp_dataset.jsonl")
        )
        self.lora_r = int(train_cfg.get("lora_r", 16))
        self.lora_alpha = int(train_cfg.get("lora_alpha", 32))
        self.lora_dropout = float(train_cfg.get("lora_dropout", 0.05))
        self.learning_rate = float(train_cfg.get("learning_rate", 2e-4))
        self.num_train_epochs = int(train_cfg.get("num_train_epochs", 3))
        self.batch_size = int(train_cfg.get("per_device_train_batch_size", 2))
        self.grad_accum = int(train_cfg.get("gradient_accumulation_steps", 8))
        self.max_seq_length = int(train_cfg.get("max_seq_length", 4096))

    @staticmethod
    def format_sft_record_as_chat(record: Dict[str, Any]) -> Dict[str, str]:
        """Format a POMDP step record into prompt-completion text pair."""
        prompt = (
            f"Utterance: {record.get('utterance', '')}\n"
            f"Context: {json.dumps(record.get('context', {}), ensure_ascii=False)}\n"
            f"Schemas: {json.dumps(record.get('pruned_schemas', []), ensure_ascii=False)}\n"
            f"History: {json.dumps(record.get('history_so_far', []), ensure_ascii=False)}\n"
            f"Step: {record.get('step_idx', 1)}\nDecision JSON:"
        )
        completion = json.dumps(record.get("target", {}), ensure_ascii=False)
        return {"prompt": prompt, "completion": completion}

    def load_sft_records(self) -> List[Dict[str, str]]:
        """Load and format all SFT records from disk."""
        path = Path(self.sft_dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"SFT dataset not found at: {path}")
        formatted: List[Dict[str, str]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    raw = json.loads(line)
                    formatted.append(self.format_sft_record_as_chat(raw))
        return formatted

    def train(self) -> Dict[str, Any]:
        """Run LoRA SFT training if base model weights and GPU/PEFT stack are available."""
        records = self.load_sft_records()
        model_path = Path(self.base_model_path)
        if not model_path.exists():
            return {
                "status": "skipped_no_local_base_model",
                "base_model_path": str(model_path),
                "num_sft_records_ready": len(records),
            }

        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )

        tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            local_files_only=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        peft_model = get_peft_model(model, peft_config)

        class _SimpleDataset(torch.utils.data.Dataset):
            def __init__(self, items: List[Dict[str, str]], tok: Any, max_len: int) -> None:
                self.items = items
                self.tok = tok
                self.max_len = max_len

            def __len__(self) -> int:
                return len(self.items)

            def __getitem__(self, idx: int) -> Dict[str, Any]:
                item = self.items[idx]
                full_text = f"{item['prompt']} {item['completion']}{self.tok.eos_token}"
                enc = self.tok(
                    full_text,
                    truncation=True,
                    max_length=self.max_len,
                    padding="max_length",
                    return_tensors="pt",
                )
                input_ids = enc["input_ids"].squeeze(0)
                attention_mask = enc["attention_mask"].squeeze(0)
                return {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": input_ids.clone(),
                }

        dataset = _SimpleDataset(records, tokenizer, self.max_seq_length)
        out_dir = Path(self.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=str(out_dir),
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.grad_accum,
            learning_rate=self.learning_rate,
            fp16=torch.cuda.is_available(),
            logging_steps=10,
            save_strategy="epoch",
            report_to=[],
        )

        trainer = Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=dataset,
        )
        trainer.train()
        peft_model.save_pretrained(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))

        return {
            "status": "completed",
            "adapter_saved_to": str(out_dir),
            "num_sft_records": len(records),
        }