import json
import re
from dotenv import load_dotenv
from openai import OpenAI
from tools import search_openalex_raw
from memory import search_memory, save_papers_to_memory

load_dotenv()

client = OpenAI()

MIN_PAPERS = 3
MAX_RETRIES = 4


def chat_text(model, system, user, temperature=0):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
    )
    return (response.choices[0].message.content or "").strip()


def extract_keywords(topic, attempt):
    system = "You generate compact academic search keywords."
    user = (
        f"Generate 3 to 5 concise keywords for: {topic}. "
        "Return only a comma-separated list."
    )
    if attempt > 0:
        user += " Focus on recent, high-impact, peer-reviewed work."
    return chat_text("gpt-4o-mini", system, user, temperature=0)


def parse_cached_papers(mem_results):
    documents = mem_results.get("documents", [[]])[0] or []
    metadatas = mem_results.get("metadatas", [[]])[0] or []
    papers = []
    for meta in metadatas:
        title = meta.get("title")
        year = meta.get("year")
        if title:
            papers.append({"id": title, "title": title, "year": year})
    if papers:
        return papers

    # Fallback: parse "Title (Year)" strings
    for doc in documents:
        match = re.match(r"^(.*)\s+\((\d{4})\)$", doc.strip())
        if match:
            papers.append(
                {"id": match.group(1), "title": match.group(1), "year": match.group(2)}
            )
    return papers


def run_critic(user_topic, papers):
    if not papers:
        return []

    system = (
        "You are a strict peer reviewer. Filter papers by relevance, recency, and "
        "signal quality. Keep only the best candidates."
    )
    user = (
        f"Topic: {user_topic}\n"
        "Return JSON with keys: approved (list of objects with id,title,year) and "
        "reason (string). Use only papers from the input list.\n\n"
        f"Input papers:\n{json.dumps(papers, ensure_ascii=True)}"
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or "{}"
    try:
        data = json.loads(content)
        approved = data.get("approved", [])
        if isinstance(approved, list):
            return approved
    except json.JSONDecodeError:
        pass
    return []

def run_scribe_agent(user_topic, validated_papers):
    """Synthesizes the final research report in Markdown."""
    print("✍️ Scribe is generating the professional report...")
    
    system = "You are a professional academic scribe. Write clear, structured Markdown."
    user = f"""
Topic: {user_topic}
Validated Papers: {validated_papers}

Task: Write a research summary in professional Markdown with:
1. Executive Summary
2. Key Findings
3. Future Outlook (2026)
4. Conclusion
5. References

Format citations as (Author, Year) when possible.
"""
    return chat_text("gpt-4o", system, user, temperature=0.2)

def run_research_pipeline(user_topic):
    # Phase 1: Memory-first search
    print(f"🧠 Checking local memory for: {user_topic}...")
    mem_results = search_memory(user_topic)
    cached_papers = parse_cached_papers(mem_results)
    validated = run_critic(user_topic, cached_papers)

    if len(validated) < MIN_PAPERS:
        print("🌐 Searching the web (OpenAlex)...")
        validated = []
        for attempt in range(MAX_RETRIES + 1):
            keywords = extract_keywords(user_topic, attempt)
            raw_web_results = search_openalex_raw(keywords, per_page=5)
            if isinstance(raw_web_results, dict) and raw_web_results.get("error"):
                print(f"OpenAlex error: {raw_web_results['error']}")
                continue
            if not raw_web_results:
                continue
            save_papers_to_memory(raw_web_results, user_topic)
            validated = run_critic(user_topic, raw_web_results)
            if len(validated) >= MIN_PAPERS:
                break

    if not validated:
        return "No high-quality papers found."

    # Phase 3: Scribe synthesis
    report = run_scribe_agent(user_topic, validated)
    
    # Save the report to a file
    with open("research_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    
    print("✅ Report saved to 'research_report.md'")
    return report

if __name__ == "__main__":
    topic = "AI Agents in Software Engineering 2026"
    print(run_research_pipeline(topic))