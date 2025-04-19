# ACS Medical Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about Acute Coronary Syndrome (ACS).

## Features

- Interactive chat interface using Streamlit
- RAG system using ChromaDB for vector storage and retrieval
- Integration with OpenAI API for language generation

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Set up your GitHub token as an environment variable:
   ```
   # On Windows
   set GITHUB_TOKEN=your_github_token_here

   # On Linux/Mac
   export GITHUB_TOKEN=your_github_token_here
   ```

3. Initialize the ChromaDB database:
   ```
   python chromadb_initializer.py
   ```

4. Run the Streamlit app:
   ```
   streamlit run chatbot_ui.py
   ```

## Usage

1. Type your question about ACS in the input field
2. Click "Ask" to get a response
3. The chat history will be displayed in reverse chronological order

## How It Works

1. User questions are embedded using Sentence Transformers
2. Similar Q&A pairs are retrieved from ChromaDB
3. Retrieved context is sent to the OpenAI API along with the user query
4. The response is displayed to the user in a chat interface