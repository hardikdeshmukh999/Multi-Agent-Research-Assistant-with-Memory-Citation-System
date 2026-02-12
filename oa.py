import requests
from urllib.parse import quote


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


def get_raw_openalex_output(query):
    clean_query = query.strip()
    encoded = quote(clean_query, safe="")
    url = f"https://api.openalex.org/works?search={encoded}&per-page=5&select=title,authorships,abstract_inverted_index"

    headers = {"User-Agent": "RawOutputScript/1.0 (mailto:your_email@example.com)"}

    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()

        for work in data.get("results", []):
            title = work.get("title", "Unknown Title")
            authorships = work.get("authorships") or []
            authors = [a.get("author", {}).get("display_name", "") for a in authorships if a.get("author")]
            authors = [a for a in authors if a]
            abstract = abstract_from_inverted_index(work.get("abstract_inverted_index") or {})

            print("=" * 60)
            print(f"Title: {title}")
            print(f"Authors: {', '.join(authors) if authors else 'N/A'}")
            print(f"Abstract: {abstract or '(not provided)'}")
            print()

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    get_raw_openalex_output("Predicting mortality rate and associated risks in COVID-19 patients")
