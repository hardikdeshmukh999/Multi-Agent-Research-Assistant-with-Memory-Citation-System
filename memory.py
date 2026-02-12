# memory.py
import logging
import os
import time
import chromadb
import numpy as np
from chromadb.utils import embedding_functions

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

# Max abstract length for storage (ChromaDB metadata + embedding doc)
# 8000 chars allows full abstracts including long structured ones
ABSTRACT_MAX_LEN = 8000

def save_papers_to_memory(papers, topic):
    """Saves a list of paper dictionaries to the local vector store."""
    if not papers: return

    ids = [str(hash(p['title'])) for p in papers]
    abstract_snippet = lambda a: ((a or "")[:ABSTRACT_MAX_LEN]).strip()
    # Document: title + year + abstract snippet for better semantic search
    documents = [
        f"{p['title']} ({p['year']}) {abstract_snippet(p.get('abstract'))}"
        for p in papers
    ]
    metadatas = [
        {
            "topic": topic,
            "title": (p.get("title") or "")[:500],  # ChromaDB metadata limit
            "year": p["year"],
            "author": p.get("author", "") or "",
            "link": p.get("link", "") or "",
            "abstract": abstract_snippet(p.get("abstract")),
        }
        for p in papers
    ]
    
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    print(f"💾 Saved {len(papers)} papers to local memory.")

logger = logging.getLogger(__name__)

# Min cosine similarity (0-1) for current query to match stored topic
TOPIC_MATCH_SIMILARITY_THRESHOLD = 0.95

def query_matches_stored_topics(query, metadatas, threshold=TOPIC_MATCH_SIMILARITY_THRESHOLD):
    """Return True if current query semantically matches any stored topic."""
    if not metadatas:
        return False
    meta_list = metadatas[0] if isinstance(metadatas[0], list) else metadatas
    topics = list(set(m.get("topic", "").strip() for m in meta_list if m and m.get("topic")))
    if not topics:
        return False
    try:
        query_embs = local_ef([query])
        topic_embs = local_ef(topics)
        q = np.array(query_embs[0], dtype=float)
        for te in topic_embs:
            t = np.array(te, dtype=float)
            sim = np.dot(q, t) / (np.linalg.norm(q) * np.linalg.norm(t) + 1e-9)
            if sim >= threshold:
                return True
    except Exception as e:
        logger.warning(f"Topic match check failed: {e}")
        return False
    return False


def flush_memory():
    """Delete all papers from the ChromaDB collection."""
    try:
        ids = collection.get()["ids"]
        if ids:
            collection.delete(ids=ids)
            print(f"🗑️ Flushed {len(ids)} papers from local memory.")
            return len(ids)
        print("🗑️ Memory already empty.")
        return 0
    except Exception as e:
        logger.error(f"Flush memory failed: {e}")
        raise


def search_memory(query, n_results=3, max_retries=3):
    """Search memory with retry logic and error handling."""
    for attempt in range(max_retries):
        try:
            results = collection.query(
                query_texts=[query],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )
            
            # Validate results structure
            if not results or 'documents' not in results:
                logger.warning(f"Invalid memory results for: {query}")
                return None
            
            return results
            
        except Exception as e:
            logger.error(f"Memory search failed (attempt {attempt + 1}): {e}")
            if attempt == max_retries - 1:
                logger.error("All retry attempts exhausted")
                return None
            time.sleep(0.5 * (attempt + 1))  # Exponential backoff
    
    return None