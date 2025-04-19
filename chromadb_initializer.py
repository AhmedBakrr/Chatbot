# chromadb_initializer.py

import chromadb
from sentence_transformers import SentenceTransformer
import uuid
import re

# Load the text file and extract Q&A pairs
def extract_qa_from_text(text_path):
    with open(text_path, 'r') as f:
        text = f.read()

    # Extract Q&A pairs with regex
    qa_pairs = re.findall(r"Q: (.*?)\nA: (.*?)(?=\nQ: |\Z)", text, re.DOTALL)
    return [{"question": q.strip(), "answer": a.strip()} for q, a in qa_pairs]

# Initialize ChromaDB (new API)
chroma_client = chromadb.PersistentClient(path="./chroma_store")

# Create or get a collection
collection_name = "acs_qa"
if collection_name in [c.name for c in chroma_client.list_collections()]:
    collection = chroma_client.get_collection(collection_name)
else:
    collection = chroma_client.create_collection(collection_name)

# Load Q&A from text file
text_path = "ACS_QA_dataset.txt"
qa_data = extract_qa_from_text(text_path)

# Embedder
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Add to Chroma
for item in qa_data:
    text = f"Q: {item['question']}\nA: {item['answer']}"
    embedding = embedder.encode(text).tolist()
    collection.add(
        documents=[text],
        embeddings=[embedding],
        ids=[str(uuid.uuid4())],
        metadatas=[{"question": item["question"]}]
    )

print(f"✅ {len(qa_data)} Q&A pairs added to ChromaDB.")

