from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_model(model_name: str = "BAAI/bge-small-en-v1.5") -> HuggingFaceEmbeddings:
    """
    Returns a configured HuggingFace embedding instance.
    """
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}  # Standardizes for Cosine Similarity
    )