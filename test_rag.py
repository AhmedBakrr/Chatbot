#!/usr/bin/env python3
"""
Test script to verify the RAG system is working correctly.
Run this after initializing the ChromaDB database.
"""

from ai_service import ai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check if GitHub token is set
if not os.environ.get("GITHUB_TOKEN"):
    print("❌ ERROR: GITHUB_TOKEN environment variable not set.")
    print("Please set your GitHub token in the .env file or environment variables.")
    exit(1)

# Test queries
test_queries = [
    "What is ACS?",
    "What symptoms should I watch for with ACS?",
    "How do doctors diagnose ACS?",
]

print("🔍 Testing RAG system with sample queries...")
print("-" * 50)

for query in test_queries:
    print(f"Query: {query}")
    try:
        response = ai.ask(query)
        print(f"Response: {response}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    print("-" * 50)

print("✅ Test completed!") 