import os
import requests
from urllib.parse import quote
from dotenv import load_dotenv
from crewai.tools import tool
from memory import save_papers_to_memory, search_memory, query_matches_stored_topics
from source_tracker import append_event

load_dotenv()


def abstract_from_inverted_index(abstract_inverted_index):
    """
    Convert OpenAlex abstract_inverted_index (word -> positions) to plain text.
    Returns empty string if missing or invalid.
    """
    if not abstract_inverted_index or not isinstance(abstract_inverted_index, dict):
        return ""
    pairs = []
    for word, positions in abstract_inverted_index.items():
        for pos in positions:
            pairs.append((pos, word))
    pairs.sort(key=lambda x: x[0])
    return " ".join(w for _, w in pairs)

@tool("OpenAlex Search")
def search_openalex(query: str):
    """
    Academic Search Tool. 
    Input: A space-separated string of keywords (e.g. "transfer learning xception").
    Avoids natural language stop words like "using", "for", "with".
    """
    # 1. WORLD-CLASS CLEANING (The "Safety Net")
    # Even if the agent sends "using xception", we strip it here just in case.
    stop_words = [" using ", " with ", " for ", " in ", " the ", " an ", " on ", " based on ", " approach "]
    clean_query = query
    for word in stop_words:
        # Case-insensitive replacement
        clean_query = clean_query.replace(word, " ").replace(word.upper(), " ").replace(word.title(), " ")
    
    # Remove double spaces created by replacement
    search_query = " ".join(clean_query.split())

    append_event(["tool_called", "search_openalex", search_query])
    
    print("=" * 60)
    print("🔧 TOOL CALLED: search_openalex")
    print(f"📥 RAW INPUT: {query}")
    print(f"✨ REFINED QUERY: {search_query}") # Logs the "Magic" to the user
    print("=" * 60)
    # Use query exactly as provided — do not rewrite or modify
    search_query = query.strip() if query else ""
    
    # --- PHASE 1: CHECK LOCAL MEMORY (RAG) ---
    try:
        # Search for top 3 matching papers in our database
        mem_results = search_memory(search_query, n_results=25)
        
        # Check if we actually found documents
        if mem_results and mem_results.get('documents') and len(mem_results['documents'][0]) > 0:
            count = len(mem_results['documents'][0])
            
            # Check 1: current query must match the stored topic (previous user's query)
            if not query_matches_stored_topics(search_query, mem_results.get("metadatas", [])):
                append_event(["mem_fallback", search_query])
                print(f"🧠 Found {count} in Memory but query does not match stored topic — falling through to OpenAlex.")
            # Check 2: memory results must be semantically relevant (cosine distance threshold)
            elif (
                (distances := mem_results.get("distances", [[]])[0])
                and (best_distance := min(distances) if distances else float("inf")) > (MEMORY_DISTANCE_THRESHOLD := 0.9)
            ):
                append_event(["mem_fallback", search_query])
                print(f"🧠 Found {count} in Memory but best match distance {best_distance:.2f} > {MEMORY_DISTANCE_THRESHOLD} — falling through to OpenAlex.")
            else:
                append_event("mem")
                print(f"🧠 Found {count} relevant papers in Local Memory.")
                formatted_mem = []
                for i, doc in enumerate(mem_results['documents'][0]):
                    meta = mem_results['metadatas'][0][i]
                    title = meta.get("title", "").strip() or doc.split("\n")[0].split(" (")[0]
                    abstract = meta.get("abstract", "").strip()
                    formatted_mem.append(f"Title: {title}\nYear: {meta.get('year', 'N/A')}\nAuthor: {meta.get('author', 'N/A')}\nLink: {meta.get('link', 'N/A')}\nAbstract: {abstract or '(not provided)'}\n(Source: Memory)")
                print("=" * 60)
                print(f"🔧 TOOL RETURNING: {len(formatted_mem)} results (from Memory)")
                print("=" * 60)
                return "\n\n".join(formatted_mem) if formatted_mem else "No papers found."
            
    except Exception as e:
        # If memory fails, just print a warning and continue to API
        print(f"⚠️ Memory check failed: {e}")

    # --- PHASE 2: SEARCH OPENALEX API ---
    # [UI HOOK] Log query for summary (file-based; works across subprocesses)
    append_event("web")
    append_event(["openalex_query", search_query])
    print(f"📋 OPENALEX_QUERY: {search_query}")
    print(f"🌐 Searching OpenAlex API for: '{search_query}'...")
    
    api_key = os.getenv("OPENALEX_API_KEY")
    # URL-encode the search string (preserves query semantics; required for HTTP)
    encoded_search = quote(search_query, safe="")
    url = f"https://api.openalex.org/works?search={encoded_search}&per-page=5&select=title,publication_year,doi,id,authorships,abstract_inverted_index"
    
    # Use your email for the "Polite Pool" (faster response times)
    headers = {"User-Agent": "AI-Researcher/1.0 (mailto:your_email@example.com)"}
    if api_key:
        url += f"&mailto={api_key}" # Some endpoints prefer api_key in param
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        results = []
        papers_to_save = []
        
        for work in data.get('results', []):
            title = work.get("title", "Unknown Title")
            year = work.get("publication_year", "N/A")
            # Prefer DOI, fallback to OpenAlex ID
            link = work.get("doi") if work.get("doi") else work.get("id")
            # First author from authorships (each has author.display_name)
            authorships = work.get("authorships") or []
            first = authorships[0] if authorships else {}
            author = (first.get("author") or {}).get("display_name", "") or ""
            # Abstract from API (abstract_inverted_index -> plain text for Critic to check relevance)
            abstract = abstract_from_inverted_index(work.get("abstract_inverted_index") or {})
            
            # 1. Format for Agent (Readable) — include Abstract so Critic can validate query vs abstract
            results.append(f"Title: {title}\nYear: {year}\nAuthor: {author}\nLink: {link}\nAbstract: {abstract or '(not provided)'}\n---")
            
            # 2. Format for Memory (Structured) — include abstract for Critic when served from cache
            papers_to_save.append({
                "title": title,
                "year": str(year),
                "author": author,
                "link": link,
                "abstract": abstract,
            })
            
        # --- PHASE 3: SAVE TO MEMORY ---
        if papers_to_save:
            save_papers_to_memory(papers_to_save, search_query)
        
        print(f"📋 OPENALEX_RESULTS: {len(results)} papers returned for query: {search_query}")
        return "\n".join(results) if results else "No papers found."
# claude code improvement
    except requests.exceptions.RequestException as e:
        return f"🚨 API Error: {e}"
    
    papers_with_scores = []
    for work in data.get('results', []):
        title = work.get("title", "Unknown")
        year = work.get("publication_year", 0)
        citations = work.get("cited_by_count", 0)
        
        # Calculate quality score
        recency_score = 1.0 if year >= 2024 else (0.5 if year >= 2020 else 0.2)
        citation_score = min(citations / 100, 1.0)  # Normalize to 0-1
        quality_score = (recency_score * 0.6) + (citation_score * 0.4)
        
        authorships = work.get("authorships") or []
        first = authorships[0] if authorships else {}
        author = (first.get("author") or {}).get("display_name", "") or ""
        abstract = abstract_from_inverted_index(work.get("abstract_inverted_index") or {})
        papers_with_scores.append({
            "title": title,
            "year": year,
            "author": author,
            "link": work.get("doi") or work.get("id"),
            "abstract": abstract,
            "citations": citations,
            "quality_score": round(quality_score, 2),
        })
        # Sort by quality score
        # 5. SORT & TRUNCATE (keep top 10)
        papers_with_scores.sort(key=lambda x: x['quality_score'], reverse=True)
        top_papers = papers_with_scores[:10]
        
        # --- NEW STEP: FILTER OUT PAPERS WITHOUT ABSTRACTS BEFORE SAVING ---
        high_quality_papers_for_memory = [
            p for p in top_papers if p['abstract'] and p['abstract'] != "(not provided)"
        ]
        
        # 6. SAVE TO MEMORY
        if high_quality_papers_for_memory:
            save_papers_to_memory(high_quality_papers_for_memory, search_query)
    
    # Return structured JSON (not strings)
    return {
        "papers": papers_with_scores[:10],  # Top 10
        "total_found": len(papers_with_scores),
        "average_quality": sum(p['quality_score'] for p in papers_with_scores) / len(papers_with_scores)
    }
    