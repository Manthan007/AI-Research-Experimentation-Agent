import chromadb
from typing import List, Dict, Any
from src.REA.retrieval.embeddings import get_embedding_model

class VectorStoreManager:
    def __init__(self, persist_directory: str = "data/vector_db"):
        self.embedding_fn = get_embedding_model()
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name="arxiv_papers",
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: List[Dict[str, Any]]):
        """
        Extracts texts, ids, and metadata from chunks list 
        and adds them to ChromaDB.
        """
        # 1. Prepare lists for Chroma format: ids, documents (texts), and metadatas
        ids = [c["chunk_id"] for c in chunks]
        documents = [c["text"] for c in chunks]
        
        # Chroma metadata dictionary MUST be flat (no nested dicts/lists)
        metadatas = [
            {
                "paper_id": c["paper_id"],
                "page_number": c["page_number"],
                "title": c["metadata"]["title"],
                "published_year": c["metadata"]["published_year"]
            }
            for c in chunks
        ]

        # 2. Embed and store
        embeddings = self.embedding_fn.embed_documents(documents)
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

    def query(self, query_text: str, top_k: int = 3, filter_dict: dict = None) -> dict:
        """Embeds query text and returns top_k most similar chunks."""
        query_vector = self.embedding_fn.embed_query(query_text)
        
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=filter_dict 
        )
        return results