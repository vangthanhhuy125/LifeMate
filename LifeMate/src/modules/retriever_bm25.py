class AppOverlapRetriever:
    def __init__(self, top_k=3):
        self.top_k = top_k
        self.corpus = []
    def build(self, data):
        self.corpus = data
    def retrieve(self, query, installed_apps, exclude_ids=None):
        exclude_ids = exclude_ids or set()
        # Mock BM25 retrieval with hard filtering
        res = [c for c in self.corpus if c['id'] not in exclude_ids][:self.top_k]
        return res