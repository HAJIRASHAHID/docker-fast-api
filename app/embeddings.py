  # Embedding creation and Pinecone upload
import numpy as np
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

def create_embeddings(chunks: list, batch_size: int = 64):
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    # batch encode to avoid using too much memory at once
    chunk_embeddings = embedding_model.encode(
        chunks,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True
    )
    return np.array(chunk_embeddings).astype("float32"), embedding_model

def init_pinecone(api_key: str, index_name: str):
    try:
        pc = Pinecone(api_key=api_key)
        return pc.Index(index_name)
    except Exception as e:
        print("Pinecone error:", e)
        return None

def upsert_chunks(index, chunks, chunk_embeddings, batch_size=50):
    if chunk_embeddings is None or len(chunk_embeddings) == 0:
        return

    total = len(chunk_embeddings)
    for i in range(0, total, batch_size):
        batch_vectors = []
        for j in range(i, min(i + batch_size, total)):
            emb = chunk_embeddings[j]
            batch_vectors.append({
                "id": str(j),
                "values": emb.tolist(),
                "metadata": {"text": chunks[j]}
            })
        index.upsert(vectors=batch_vectors)
