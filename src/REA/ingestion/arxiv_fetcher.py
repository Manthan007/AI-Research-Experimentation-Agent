import arxiv
import logging
import time

logger = logging.getLogger(__name__)

def fetch_arxiv_papers(query: str, max_results: int = 5, delay_seconds: float = 3.0) -> list[dict]:

    client = arxiv.Client(
        page_size=max_results,
        delay_seconds=delay_seconds,
        num_retries=5
    )

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    papers = []
    
    try:
        for result in client.results(search):
            paper_data = {
                "paper_id": result.entry_id.split("/")[-1],  
                "title": result.title,
                "authors": [author.name for author in result.authors],
                "published_year": result.published.year,
                "pdf_url": result.pdf_url,
            }
            papers.append(paper_data)
            
    except arxiv.HTTPError as e:
        logger.error(f"arXiv HTTP error encountered: {e}")
        if e.status == 429:
            logger.warning("Rate limit hit! Waiting 10 seconds before continuing...")
            time.sleep(10)
    except Exception as e:
        logger.error(f"Unexpected error fetching papers from arXiv: {e}")
        
    return papers