import os
import requests
from urllib.parse import quote
from dotenv import load_dotenv
from crewai.tools import tool
from memory import save_papers_to_memory, search_memory, query_matches_stored_topics
from source_tracker import append_event

load_dotenv()

def abstract_from_inverted_index(abstract_inverted_index):
    """Convert OpenAlex abstract_inverted_index to plain text."""
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
    Input: Keywords (e.g. "transfer learning xception"). Strips stop words automatically.
    """
    # 1. QUERY CLEANING
    stop_words = [" using ", " with ", " for ", " in ", " the ", " an ", " on ", " based on ", " approach "]
    clean_query = query
    for word in stop_words:
        clean_query = clean_query.replace(word, " ").replace(word.upper(), " ").replace(word.title(), " ")
    
    search_query = " ".join(clean_query.split()).strip()
    append_event(["tool_called", "search_openalex", search_query])

    # 2. CHECK LOCAL MEMORY (RAG)
    try:
        mem_results = search_memory(search_query, n_results=3)
        if mem_results and mem_results.get('documents') and len(mem_results['documents'][0]) > 0:
            if query_matches_stored_topics(search_query, mem_results.get("metadatas", [])):
                append_event("mem")
                formatted_mem = []
                for i, doc in enumerate(mem_results['documents'][0]):
                    meta = mem_results['metadatas'][0][i]
                    formatted_mem.append(
                        f"Title: {meta.get('title')}\n"
                        f"Year: {meta.get('year')}\n"
                        f"Author: {meta.get('author')}\n"
                        f"Link: {meta.get('link')}\n"
                        f"Abstract: {meta.get('abstract') or '(not provided)'}\n"
                        f"(Source: Memory)"
                    )
                return "\n\n".join(formatted_mem)
    except Exception as e:
        print(f"⚠️ Memory check failed: {e}")

    # 3. API SEARCH (DEEP SWEEP)
    append_event("web")
    encoded_search = quote(search_query, safe="")
    url = f"https://api.openalex.org/works?search={encoded_search}&per-page=50&select=title,publication_year,doi,id,authorships,abstract_inverted_index,cited_by_count"
    
    headers = {"User-Agent": "AI-Researcher/1.0 (mailto:your_email@example.com)"}
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()
        
        # 4. QUALITY SCORING & FILTERING
        papers_with_scores = []
        for work in data.get('results', []):
            title = work.get("title", "Unknown")
            year = work.get("publication_year", 0)
            citations = work.get("cited_by_count", 0)
            
            # Recency (70%) + Citations (30%)
            recency_score = 1.0 if year >= 2024 else (0.5 if year >= 2020 else 0.2)
            citation_score = min(citations / 100, 1.0)
            quality_score = (recency_score * 0.7) + (citation_score * 0.3)
            
            authorships = work.get("authorships") or []
            author = (authorships[0].get("author") or {}).get("display_name", "Unknown") if authorships else "Unknown"
            abstract = abstract_from_inverted_index(work.get("abstract_inverted_index"))

            papers_with_scores.append({
                "title": title,
                "year": str(year),
                "author": author,
                "link": work.get("doi") or work.get("id"),
                "abstract": abstract,
                "quality_score": round(quality_score, 2)
            })

        # 5. SORT & TRUNCATE
        papers_with_scores.sort(key=lambda x: x['quality_score'], reverse=True)
        top_papers = papers_with_scores[:10]
        
        # 6. SAVE TO MEMORY
        if top_papers:
            save_papers_to_memory(top_papers, search_query)
        
        # 7. FORMAT FOR DR. KIMI (The Scribe)
        final_output = []
        for p in top_papers:
            final_output.append(
                f"Title: {p['title']}\n"
                f"Year: {p['year']}\n"
                f"Author: {p['author']}\n"
                f"Link: {p['link']}\n"
                f"Quality Score: {p['quality_score']}\n"
                f"Abstract: {p['abstract'] or '(not provided)'}\n"
                f"---"
            )
            
        return "\n".join(final_output) if final_output else "No papers found."

    except Exception as e:
        return f"🚨 API Error: {e}"