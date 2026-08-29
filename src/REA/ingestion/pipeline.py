from src.REA.ingestion.arxiv_fetcher import fetch_arxiv_papers
from src.REA.ingestion.pdf_parser import download_pdf, parse_pdf_text

def run_ingestion_pipeline(query: str, max_results: int = 5) -> list[dict]:
    papers = fetch_arxiv_papers(query=query, max_results=max_results)
    processed_papers = []

    for paper in papers:
        pdf_filename = f"data/raw_pdfs/{paper['paper_id']}.pdf"
        
        print(f"Downloading {paper['title']}...")
        if download_pdf(paper['pdf_url'], pdf_filename):
            print(f"Parsing {pdf_filename}...")
            extracted_pages = parse_pdf_text(pdf_filename)
            
            paper['pages'] = extracted_pages
            paper['total_pages'] = len(extracted_pages)
            processed_papers.append(paper)

    return processed_papers

if __name__ == "__main__":
    results = run_ingestion_pipeline("retrieval augmented generation", max_results=5)
    print(f"Successfully processed {len(results)} papers.")