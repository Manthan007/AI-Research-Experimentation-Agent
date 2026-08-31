from rank_bm25 import BM25Okapi
from typing import List, Dict, Any
from src.REA.retrieval.vector_store import VectorStoreManager

class HybridRetriever:
    def __init__(self, vector_store: VectorStoreManager):
        self.vector_store = vector_store
        self.bm25 = None
        self.corpus_chunks = []

    def index_chunks(self, chunks: List[Dict[str, Any]]):
        """Indexes chunks into both ChromaDB and BM25."""
        self.corpus_chunks = chunks
        
        # 1. Index into Vector Store
        self.vector_store.add_chunks(chunks)
        
        # 2. Tokenize text for BM25 (simple space/lowercase split)
        tokenized_corpus = [c["text"].lower().split(" ") for c in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _reciprocal_rank_fusion(
        self, 
        vector_results: List[str], 
        bm25_results: List[str], 
        k: int = 60
    ) -> List[str]:
        """Calculates RRF score for documents across both rank lists."""
        scores = {}

        # Process Vector search rankings
        for rank, chunk_id in enumerate(vector_results):
            if chunk_id not in scores:
                scores[chunk_id] = 0.0
            scores[chunk_id] += 1.0 / (k + rank + 1)

        # Process BM25 rankings
        for rank, chunk_id in enumerate(bm25_results):
            if chunk_id not in scores:
                scores[chunk_id] = 0.0
            scores[chunk_id] += 1.0 / (k + rank + 1)

        # Sort chunk IDs by highest combined RRF score
        sorted_chunks = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [chunk_id for chunk_id, score in sorted_chunks]

    def search(self, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Executes Hybrid Search combining Dense Vector and Sparse BM25."""
        # 1. Vector Search
        v_raw = self.vector_store.query(query_text, top_k=top_k * 2)
        vector_ids = v_raw["ids"][0] if v_raw["ids"] else []

        # 2. BM25 Search
        tokenized_query = query_text.lower().split(" ")
        bm25_top_indices = self.bm25.get_top_n(tokenized_query, range(len(self.corpus_chunks)), n=top_k * 2)
        bm25_ids = [self.corpus_chunks[idx]["chunk_id"] for idx in bm25_top_indices]

        # 3. Fuse Rankings with RRF
        fused_ids = self._reciprocal_rank_fusion(vector_ids, bm25_ids)[:top_k]

        # Return full chunk objects matching the top fused IDs
        chunk_map = {c["chunk_id"]: c for c in self.corpus_chunks}
        return [chunk_map[cid] for cid in fused_ids if cid in chunk_map]