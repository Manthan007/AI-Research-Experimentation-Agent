import os
import requests
import pymupdf  
import logging

logger = logging.getLogger(__name__)

def download_pdf(pdf_url: str, save_path: str) -> bool:
    """Downloads a PDF from a URL and saves it locally."""
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        response = requests.get(pdf_url, timeout=15)
        response.raise_for_status()

        with open(save_path, "wb") as f:
            f.write(response.content)
        return True
    except Exception as e:
        logger.error(f"Failed to download PDF from {pdf_url}: {e}")
        return False

def parse_pdf_text(pdf_path: str) -> list[dict]:
    """
    Extracts structured text page-by-page from a local PDF.
    Returns a list of dicts: [{'page_num': 1, 'text': '...'}, ...]
    """
    pages_content = []
    try:
        doc = pymupdf.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text")  # Extracts clean plain text
            
            # Basic cleanup: omit completely empty pages
            if text.strip():
                pages_content.append({
                    "page_num": page_num + 1,
                    "text": text.strip()
                })
        doc.close()
    except Exception as e:
        logger.error(f"Error parsing PDF at {pdf_path}: {e}")

    return pages_content