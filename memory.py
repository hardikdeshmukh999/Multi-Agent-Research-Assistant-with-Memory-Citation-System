# memory.py
import chromadb
from chromadb.utils import embedding_functions
import os

# 1. Setup Local Embeddings (Free & Fast)
local_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# 2. Setup ChromaDB
# This saves the database to a folder named "research_db"
client = chromadb.PersistentClient(path="./research_db")

# 3. Get or Create Collection
collection = client.get_or_create_collection(
    name="openai_research_vault", 
    embedding_function=local_ef
)

def save_papers_to_memory(papers, topic):
    """Saves a list of paper dictionaries to the local vector store."""
    if not papers: return

    # Generate unique IDs based on the paper title (simple hash)
    ids = [str(hash(p['title'])) for p in papers]
    documents = [f"{p['title']} ({p['year']})" for p in papers]
    metadatas = [{"topic": topic, "year": p['year']} for p in papers]
    
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    print(f"💾 Saved {len(papers)} papers to local memory.")

def search_memory(query, n_results=3):
    """Searches the local vector store for relevant papers."""
    return collection.query(query_texts=[query], n_results=n_results)