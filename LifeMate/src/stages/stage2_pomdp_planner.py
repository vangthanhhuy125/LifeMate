import os
import re
import json
import torch
from typing import Dict, Any, List, Optional
from src.utils.pomdp_types import Stage1OutputBundle, POMDPAction, POMDPTransitionStep

class Stage2_POMDPPolicyPlanner:
    """
    ================================================================================
    STAGE 2: CONSTRAINED POMDP POLICY PLANNER (SLM < 5B)
    - Tại mỗi bước k, nhận trạng thái niềm tin từ Stage 1 + lịch sử tương tác H_{k-1}
      (bao gồm cả phản hồi lỗi từ màng lọc Validate-Before-Call nếu có) để sinh ra
      hành động a_k thuộc A = {Call(tool, theta), AskMissing(tool, params), EmptySet}.
    ================================================================================
    """
    def __init__(self, stage2_cfg: Dict[str, Any]):
        self.cfg = stage2_cfg
        self.llm_cfg = stage2_cfg.get("llm", {})
        self.base_model_path = self.llm_cfg.get("base_model_path", "Qwen/Qwen2.5-3B-Instruct")
        self.lora_path = self.llm_cfg.get("lora_adapter_path", "")
        self.device = self.llm_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = self.llm_cfg.get("max_new_tokens", 512)

        prompt_file = stage2_cfg.get("system_prompt_path", "configs/prompts/pomdp_policy_prompt.txt")
        if os.path.exists(prompt_file):
            with open(prompt_file, "r", encoding="utf-8") as f:
                self.system_prompt = f.read().strip()
        else:
            self.system_prompt = "Bạn là POMDP Policy Agent điều khiển công cụ theo JSON Schema."

        self.tokenizer = None
        self.model = None

    def _lazy_load_llm(self):
        if self.model is None:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from peft import PeftModel
            dtype = torch.bfloat16 if self.llm_cfg.get("torch_dtype") == "bfloat16" and self.device == "cuda" else torch.float16
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_path, trust_remote_code=True)
            base_m = AutoModelForCausalLM.from_pretrained(
                self.base_model_path,
                torch_dtype=dtype,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            if self.lora_path and os.path.exists(self.lora_path):
                self.model = PeftModel.from_pretrained(base_m, self.lora_path)
            else:
                self.model = base_m
            self.model.eval()

    def format_pomdp_state_prompt(
        self,
        bundle: Stage1OutputBundle,
        history_H: List[POMDPTransitionStep],
        validation_error_feedback: Optional[str] = None
    ) -> str:
        ctx_dict = {
            "current_time_t": bundle.context_C.current_time,
            "timezone": bundle.context_C.timezone,
            "location_l": bundle.context_C.current_location,
            "user_profile_P": bundle.context_C.user_profile,
            "installed_apps_K": list(bundle.context_C.installed_apps.keys())
        }

        rag_section = ""
        if bundle.few_shot_trajectories:
            rag_section = "### QUỸ ĐẠO MẪU TƯƠNG ĐỒNG (FEW-SHOT RAG TRAJECTORIES):\n"
            for idx, ex in enumerate(bundle.few_shot_trajectories, 1):
                rag_section += (
                    f"- Mẫu {idx} [Apps: {ex['installed_apps']}]: "
                    f"Khẩu lệnh: \"{ex['utterance']}\" -> Chuỗi lệnh chuẩn: "
                    f"{json.dumps(ex['gold_tool_calls'], ensure_ascii=False)}\n"
                )

        h_trace = [
            {"step": s.step_k, "action": s.action, "observation_o_k": s.observation}
            for s in history_H
        ]

        error_block = ""
        if validation_error_feedback:
            error_block = (
                f"\n### [CẢNH BÁO TỪ MÀNG LỌC VALIDATE-BEFORE-CALL]:\n"
                f"Hành động dự kiến vừa rồi của bạn bị từ chối bởi Schema Validator với lỗi:\n"
                f"\"{validation_error_feedback}\"\n"
                f"Hãy sửa lại tên tham số/kiểu dữ liệu, hoặc gọi tool `read` để lấy đúng ID `x-ref`, "
                f"hoặc chuyển sang `ASK_MISSING` nếu thực sự thiếu thông tin từ người dùng!\n"
            )

        prompt = (
            f"{rag_section}\n"
            f"### NGỮ CẢNH BAN ĐẦU C = (t, l, P, K):\n{json.dumps(ctx_dict, ensure_ascii=False)}\n\n"
            f"### LỊCH SỬ HỘI THOẠI TRƯỚC ĐÓ:\n{json.dumps(bundle.dialogue_history, ensure_ascii=False)}\n\n"
            f"### LƯỢC ĐỒ CÔNG CỤ ĐÃ CẮT TỈA S(K):\n{json.dumps(bundle.pruned_schema_SK, ensure_ascii=False)}\n\n"
            f"### LỊCH SỬ TƯƠNG TÁC MÔI TRƯỜNG H_{{k-1}}:\n{json.dumps(h_trace, ensure_ascii=False)}\n"
            f"{error_block}\n"
            f"### KHẨU LỆNH GIỌNG NÓI HIỆN TẠI (A):\n\"{bundle.utterance_A}\"\n\n"
            f"Hãy sinh quyết định hành động a_k tiếp theo dưới dạng JSON duy nhất:"
        )
        return prompt

    def sample_action(
        self,
        bundle: Stage1OutputBundle,
        history_H: List[POMDPTransitionStep],
        validation_error_feedback: Optional[str] = None
    ) -> POMDPAction:
        self._lazy_load_llm()
        user_prompt = self.format_pomdp_state_prompt(bundle, history_H, validation_error_feedback)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        chat_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([chat_text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=self.tokenizer.eos_token_id
            )
        gen_ids = out_ids[0][len(inputs.input_ids[0]):]
        raw_output = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        return self.parse_raw_decision(raw_output)

    @staticmethod
    def parse_raw_decision(raw_text: str) -> POMDPAction:
        parsed = None
        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', raw_text, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                except json.JSONDecodeError:
                    parsed = None

        if not isinstance(parsed, dict):
            return POMDPAction(action_type="TERMINATE", thought="JSON parse fallback -> EmptySet")

        act_type = str(parsed.get("action_type", parsed.get("action", "TERMINATE"))).upper()
        thought = str(parsed.get("thought", ""))

        if "tool_call" in parsed and isinstance(parsed["tool_call"], dict):
            tc = parsed["tool_call"]
            t_name = tc.get("name", "")
            args = tc.get("arguments", {})
        else:
            t_name = parsed.get("tool_name", "")
            args = parsed.get("arguments", {})

        if act_type == "ASK_MISSING" or t_name == "sys.ask_missing_info":
            intended_tool = args.get("tool_name", t_name) if t_name == "sys.ask_missing_info" else t_name
            m_params = parsed.get("missing_params", args.get("params", []))
            return POMDPAction(
                action_type="ASK_MISSING",
                thought=thought,
                tool_name=intended_tool,
                missing_params=list(m_params) if isinstance(m_params, list) else []
            )

        if act_type == "CALL" and t_name:
            return POMDPAction(
                action_type="CALL",
                thought=thought,
                tool_name=t_name,
                arguments=args if isinstance(args, dict) else {}
            )

        return POMDPAction(action_type="TERMINATE", thought=thought)
