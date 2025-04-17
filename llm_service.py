# llm_service.py (modified for local model)
from typing import List
from transformers import pipeline

class LLMService:
    def __init__(self, model="google/flan-t5-base"): 
        # Load the model
        self.pipe = pipeline(
            "text2text-generation",
            model=model,
            max_length=512
        )

    def generate_answer(self, query: str, retrieved_docs: List[dict]) -> str:
        context = "\n\n".join([doc["content"] for doc in retrieved_docs])

        prompt = (
            "You are a medical assistant chatbot. Your ONLY job is to provide medically accurate answers related to Acute Coronary Syndrome (ACS), its symptoms, diagnosis, and treatment.\n\n"
            "DO NOT answer unrelated questions. If the question is out of scope, respond with: 'I can only help with ACS-related inquiries.'\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer the question clearly and concisely based on the provided information."
        )


        # Generate response
        response = self.pipe(prompt)[0]["generated_text"]
        return response