# rag_service.py

from sentence_transformers import SentenceTransformer
import chromadb
from typing import List

class RAGRetriever:
    def __init__(self, db_path="./chroma_store", collection_name="acs_qa"):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection(collection_name)

    def retrieve(self, query: str, top_k: int = 3) -> List[dict]:
        query_embedding = self.embedder.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas"]
        )

        # Return a list of Q&A dicts
        retrieved = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            retrieved.append({
                "question": meta.get("question", "Unknown"),
                "content": doc
            })

        return retrieved
