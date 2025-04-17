# ai_manager.py

from rag_service import RAGRetriever
from llm_service import LLMService

class AIManager:
    def __init__(self):
        self.retriever = RAGRetriever()
        self.llm = LLMService()

    def ask(self, query: str, top_k: int = 3) -> str:
        retrieved = self.retriever.retrieve(query, top_k=top_k)
        answer = self.llm.generate_answer(query, retrieved)
        return answer