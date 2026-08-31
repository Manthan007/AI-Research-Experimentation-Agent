from dotenv import load_dotenv

load_dotenv()

import os
import logging
from typing import List, Dict, Any

from src.REA.ingestion.arxiv_fetcher import fetch_arxiv_papers
from src.REA.ingestion.pdf_parser import download_pdf, parse_pdf_text
from src.REA.chunking.recursive import chunk_paper_pages
from src.REA.retrieval.vector_store import VectorStoreManager
from src.REA.retrieval.hybrid import HybridRetriever
from src.REA.retrieval.query_expander import QueryExpander

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(
        self,
        storage_dir: str = "data/pdfs",
        db_directory: str = "data/vector_db"
    ):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # 1. Initialize Vector Store, Hybrid Retriever, and Query Expander
        logger.info("Initializing Vector Store and Retriever modules...")
        self.vector_store = VectorStoreManager(persist_directory=db_directory)
        self.retriever = HybridRetriever(vector_store=self.vector_store)
        self.query_expander = QueryExpander()

    def ingest_arxiv_papers(self, query: str, max_results: int = 3) -> None:
        """
        Full Data Pipeline:
        Fetch arXiv metadata -> Download PDFs -> Parse Text -> Chunk -> Index in Vector/BM25 DBs
        """
        logger.info(f"Starting ingestion pipeline for arXiv search query: '{query}'")
        
        # Step A: Fetch arXiv metadata
        papers_metadata = fetch_arxiv_papers(query=query, max_results=max_results)
        if not papers_metadata:
            logger.warning("No papers fetched from arXiv.")
            return

        all_chunks = []

        # Step B: Process each paper sequentially
        for paper in papers_metadata:
            paper_id = paper["paper_id"]
            pdf_url = paper["pdf_url"]
            pdf_path = os.path.join(self.storage_dir, f"{paper_id}.pdf")

            logger.info(f"Processing paper: {paper['title']} ({paper_id})")

            # 1. Download PDF
            download_success = download_pdf(pdf_url=pdf_url, save_path=pdf_path)
            if not download_success:
                logger.error(f"Skipping paper {paper_id} due to download failure.")
                continue

            # 2. Extract Text Page-by-Page
            pages_content = parse_pdf_text(pdf_path=pdf_path)
            if not pages_content:
                logger.warning(f"No usable text extracted from PDF {paper_id}.")
                continue

            # Assemble full paper dictionary expected by `chunk_paper_pages`
            processed_paper = {
                "paper_id": paper_id,
                "title": paper["title"],
                "authors": paper["authors"],
                "published_year": paper["published_year"],
                "pages": pages_content
            }

            # 3. Recursive Chunking
            paper_chunks = chunk_paper_pages(processed_paper=processed_paper)
            all_chunks.extend(paper_chunks)
            logger.info(f"Generated {len(paper_chunks)} text chunks for paper '{paper_id}'.")

        # Step C: Index Chunks into Hybrid Retriever (ChromaDB + BM25)
        if all_chunks:
            logger.info(f"Indexing total {len(all_chunks)} chunks into ChromaDB & BM25 index...")
            self.retriever.index_chunks(all_chunks)
            logger.info("Ingestion completed successfully.")
        else:
            logger.warning("No chunks available for indexing.")

    def run_query(self, user_query: str, top_k_per_query: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieval Pipeline:
        Expand User Query -> Perform Multi-Query Hybrid Search -> Deduplicate Results
        """
        logger.info(f"Processing user query: '{user_query}'")

        # Step A: Generate expanded search query variations using Gemini 2.5 Flash
        query_variations = self.query_expander.expand_query(user_query)
        logger.info(f"Generated {len(query_variations)} query variations: {query_variations}")

        retrieved_chunks_map = {}

        # Step B: Execute Hybrid Retrieval for every query variation
        for q in query_variations:
            results = self.retriever.search(query_text=q, top_k=top_k_per_query)
            for chunk in results:
                cid = chunk["chunk_id"]
                if cid not in retrieved_chunks_map:
                    retrieved_chunks_map[cid] = chunk

        final_chunks = list(retrieved_chunks_map.values())
        logger.info(f"Retrieved {len(final_chunks)} unique document chunks.")
        
        return final_chunks


# --- Quick Test Execution ---
if __name__ == "__main__":
    pipeline = RAGPipeline()

    # 1. Run Ingestion (Fetches papers and creates/updates vector database)
    pipeline.ingest_arxiv_papers(query="Retrieval Augmented Generation", max_results=2)

    # 2. Run Retrieval Test
    query = "What metrics are used to evaluate RAG systems?"
    retrieved_context = pipeline.run_query(user_query=query, top_k_per_query=3)

    print("\n" + "="*50)
    print("TOP RETRIEVED CHUNKS:")
    print("="*50)
    for idx, chunk in enumerate(retrieved_context, 1):
        print(f"\n[Result {idx}] Document ID: {chunk['chunk_id']}")
        print(f"Title: {chunk['metadata']['title']}")
        print(f"Text Snippet: {chunk['text'][:200]}...")