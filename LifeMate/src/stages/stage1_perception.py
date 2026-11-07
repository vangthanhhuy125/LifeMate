from src.modules.schema_pruner import DynamicSchemaPruner
from src.modules.retriever_bm25 import AppOverlapRetriever
from src.modules.nlu_normalizers import TimeNormalizer

class Stage1:
    def __init__(self, config):
        self.pruner = DynamicSchemaPruner()
        self.retriever = AppOverlapRetriever(config.get("top_k", 3))
    
    def run(self, sample, catalog):
        transcript = sample.get("audio_text", [""])[-1]
        norm_transcript = TimeNormalizer.normalize(transcript, sample["context"].get("current_time", ""))
        pruned_schema, reduction = self.pruner.prune(catalog, sample["context"]["installed_apps"])
        rag_examples = self.retriever.retrieve(norm_transcript, sample["context"]["installed_apps"])
        
        return {
            "transcript": norm_transcript,
            "schema": pruned_schema,
            "rag": rag_examples,
            "reduction": reduction
        }