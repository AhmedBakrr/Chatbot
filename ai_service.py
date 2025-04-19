import os
import chromadb
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class ACSAssistant:
    def __init__(self):
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path="./chroma_store")
        self.collection = self.chroma_client.get_collection("acs_qa")
        
        # Initialize embedder
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Initialize OpenAI client
        self.client = OpenAI(
            base_url="https://models.inference.ai.azure.com",
            api_key=os.environ.get("GITHUB_TOKEN")
        )
        
    def ask(self, query):
        """
        Process a user query using RAG:
        1. Find relevant context from ChromaDB
        2. Construct a prompt with context and query
        3. Get response from LLM
        """
        # Get query embedding
        query_embedding = self.embedder.encode(query).tolist()
        
        # Search for relevant context in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )
        
        # Extract context from results
        context = "\n\n".join(results["documents"][0]) if results["documents"] else ""
        
        # Construct the prompt with context
        system_prompt = f"""You are an assistant specialized in Acute Coronary Syndrome (ACS).
Answer questions based on the following context information. If you don't know the answer
or the information is not in the context, say so politely.

CONTEXT:
{context}"""

        # Get response from OpenAI
        response = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": query,
                }
            ],
            model="gpt-4o-mini"  # Using the model specified in the example
        )
        
        return response.choices[0].message.content

# Create a singleton instance
ai = ACSAssistant() 