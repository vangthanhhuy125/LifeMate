import re
import torch
from typing import Dict, Any

class AcousticFrontendASR:
    """
    [Contribution 3 - Multimodal Grounding]:
    Giải mã tín hiệu giọng nói tiếng Việt 16kHz (Bắc - Trung - Nam) sang văn bản
    và chuẩn hóa từ vựng trộn mã Anh-Việt (Code-Switching Lexicon Normalizer).
    """
    def __init__(self, asr_config: Dict[str, Any]):
        self.cfg = asr_config
        self.model_id = asr_config.get("model_name_or_path", "vinai/PhoWhisper-small")
        self.device = asr_config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if asr_config.get("torch_dtype") == "float16" and self.device == "cuda" else torch.float32
        self.sr = asr_config.get("sample_rate", 16000)
        self.processor = None
        self.model = None

    def _lazy_load(self):
        if self.model is None:
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True
            ).to(self.device)
            self.model.eval()

    def transcribe_waveform(self, wav_path: str) -> str:
        import librosa
        self._lazy_load()
        waveform, _ = librosa.load(wav_path, sr=self.sr, mono=True)
        inputs = self.processor(waveform, sampling_rate=self.sr, return_tensors="pt").to(self.device, dtype=self.dtype)
        with torch.no_grad():
            pred_ids = self.model.generate(
                inputs["input_features"],
                num_beams=self.cfg.get("beam_size", 1),
                do_sample=False
            )
        raw_text = self.processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()
        return self.normalize_code_switching(raw_text)

    def normalize_code_switching(self, text: str) -> str:
        if not self.cfg.get("code_switching_normalization", True):
            return text
        lexicon = self.cfg.get("lexicon_map", {})
        out = text
        for phonetic_vi, en_term in lexicon.items():
            pattern = re.compile(re.escape(phonetic_vi), re.IGNORECASE)
            out = pattern.sub(en_term, out)
        return out
