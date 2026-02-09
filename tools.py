# tools.py
import os
import requests
from dotenv import load_dotenv
from crewai.tools import tool
from memory import save_papers_to_memory  # <--- IMPORT THE MEMORY FUNCTION

load_dotenv()

@tool("OpenAlex Search")
def search_openalex(query: str):
    """
    Searches OpenAlex for academic papers.
    Returns Title, Year, and Link (DOI).
    Automatically saves results to local memory.
    """
    api_key = os.getenv("OPENALEX_API_KEY")
    clean_query = query.replace('"', '').replace("'", "").strip()
    
    # We ask for title, year, doi, and id
    url = f"https://api.openalex.org/works?search={clean_query}&per-page=5&select=title,publication_year,doi,id,abstract_inverted_index"
    if api_key:
        url += f"&mailto={api_key}"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        results = []
        papers_to_save = [] # List to hold data for memory.py

        for work in data.get('results', []):
            title = work.get("title", "No Title")
            year = work.get("publication_year", "N/A")
            # Prefer DOI link, fallback to OpenAlex ID
            link = work.get("doi") if work.get("doi") else work.get("id")
            
            # 1. Format for the Agent to read immediately
            results.append(f"Title: {title}\nYear: {year}\nLink: {link}\n---")
            
            # 2. Format for Memory Storage
            papers_to_save.append({
                "id": work.get("id"),
                "title": title,
                "year": str(year),
                "link": link
            })
            
        # --- CRITICAL FIX: ACTUAL SAVE STEP ---
        if papers_to_save:
            save_papers_to_memory(papers_to_save, query)
        # --------------------------------------

        return "\n".join(results) if results else "No papers found."
    except Exception as e:
        return f"Error: {e}"