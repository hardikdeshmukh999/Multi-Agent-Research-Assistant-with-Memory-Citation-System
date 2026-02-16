import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

def load_context():
    """
    Reads the raw tool outputs (the 'Truth') from the source tracker.
    """
    try:
        if os.path.exists(".source_events.jsonl"):
            with open(".source_events.jsonl", "r", encoding="utf-8") as f:
                events = [json.loads(line) for line in f]
            
            # Extract only the text returned by the tools (the "Context")
            context = []
            for e in events:
                if isinstance(e, list) and e[0] == "tool_called":
                    # We capture the query
                    context.append(f"Query: {e[2]}")
                # We assume the tool output was printed or logged. 
                # In a real production system, we'd log the full tool output to a file.
                # For this V1, we will use the 'Abstracts' that made it into the report references as proxy,
                # OR we rely on the fact that the agent is answering based on memory.
            return "\n".join(context)
    except Exception:
        return ""

def evaluate_faithfulness(report_text, query):
    """
    The 'Judge': Checks if the report answers the specific query 
    and if its tone matches the persona.
    """
    
    prompt = f"""
    You are an impartial Judge evaluating an AI Research Agent's report.
    
    USER QUERY: "{query}"
    
    AI REPORT:
    {report_text}
    
    ---
    
    YOUR TASK:
    Evaluate this report on two metrics (0-100 score):
    1. ANSWER RELEVANCE: Does it directly answer the user's specific query? (Or did it ramble?)
    2. PERSONA ADHERENCE: Does it sound like "Dr. Kimi" (Spicy, Emoji-using, Anime-fan PhD)?
    
    Return valid JSON only:
    {{
        "relevance_score": 95,
        "persona_score": 90,
        "critique": "The report answered the question but lacked enough 'spicy' takes."
    }}
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"relevance_score": 0, "persona_score": 0, "critique": f"Evaluation failed: {e}"}

def run_evaluation(report_path, query):
    if not os.path.exists(report_path):
        return None
        
    with open(report_path, "r", encoding="utf-8") as f:
        report_text = f.read()
        
    # Run the Judge
    scores = evaluate_faithfulness(report_text, query)
    return scores