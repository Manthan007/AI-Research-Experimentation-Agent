from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_paper_pages(
    processed_paper: dict, 
    chunk_size: int = 800, 
    chunk_overlap: int = 150
) -> list[dict]:
    
    # 1. Instantiate the RecursiveCharacterTextSplitter with separators: ["\n\n", "\n", " ", ""]
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size = chunk_size,
        chunk_overlap  = chunk_overlap)

    chunk_list = []

    paper_id = processed_paper["paper_id"]
    title = processed_paper.get("title", "")
    authors = processed_paper.get("authors", [])
    published_year = processed_paper.get("published_year", 0)
    

    # 2. Loop through each page in processed_paper["pages"]
    for page in processed_paper["pages"]:

        page_num = page["page_num"]

        text_chunk = splitter.split_text(page["text"])

        for c_idx, chunk_text in enumerate(text_chunk):
            chunk_dict = {
                "chunk_id": f"{paper_id}_p{page_num}_c{c_idx}",
                "paper_id": paper_id,
                "page_number": page_num,
                "text": chunk_text,
                "metadata": {
                    "title": title,
                    "authors": authors,
                    "published_year": published_year
                }
            }

            chunk_list.append(chunk_dict)

    return chunk_list