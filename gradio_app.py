# gradio_app.py
import gradio as gr
import sys
import threading
import time
from crew import run_crew
from memory import flush_memory

# --- 1. THE HIDDEN LOGGER ---
# This catches the print statements silently so we can count them later.
class SourceTracker:
    def __init__(self):
        self.logs = []
        
    def write(self, text):
        clean = text.strip()
        if clean:
            self.logs.append(clean)

    def flush(self): pass

    def get_summary(self, direct_events=None):
        """Build summary from print logs and/or direct events from tools."""
        mem_hits = sum(1 for line in self.logs if "🧠" in line)
        web_hits = sum(1 for line in self.logs if "🌐" in line)
        openalex_queries = [
            line.replace("📋 OPENALEX_QUERY:", "").strip()
            for line in self.logs
            if "📋 OPENALEX_QUERY:" in line
        ]
        # Prefer direct events (file-based; works when tools run in subprocess)
        if direct_events:
            mem_hits = sum(1 for e in direct_events if e == "mem")
            web_hits = sum(1 for e in direct_events if e == "web")
            openalex_queries = [e[1] for e in direct_events if isinstance(e, list) and len(e) == 2 and e[0] == "openalex_query"]
        query_section = ""
        if openalex_queries:
            unique_queries = list(dict.fromkeys(openalex_queries))
            query_section = "\n- **OpenAlex query used:** " + "; ".join(f"\"{q}\"" for q in unique_queries)
        no_tool_note = ""
        if direct_events and mem_hits == 0 and web_hits == 0:
            tool_called = any(isinstance(e, list) and e and e[0] == "tool_called" for e in direct_events)
            if not tool_called:
                no_tool_note = "\n- ⚠️ No tool calls detected — the Librarian may not have invoked the search."
        return f"""
        ### 📊 Data Source Summary
        - **Local Memory (ChromaDB):** Used {mem_hits} times (Fast & Free)
        - **External API (OpenAlex):** Used {web_hits} times (Slow & Costly)
        {query_section}{no_tool_note}
        """

# --- 2. THE RESEARCH FUNCTION ---
def run_research(topic):
    # Clear events file (works across subprocesses)
    from source_tracker import clear_events, get_events
    clear_events()
    
    # Setup the silent tracker (backup for print-based capture)
    tracker = SourceTracker()
    original_stdout = sys.stdout
    sys.stdout = tracker  # Redirect print() to our tracker
    
    # Run AI in background
    result_container = {"data": None}
    def target():
        try:
            result_container["data"] = str(run_crew(topic))
        except Exception as e:
            result_container["data"] = f"Error: {e}"
    
    thread = threading.Thread(target=target)
    thread.start()
    
    # --- UI UPDATE LOOP ---
    # While working, just show one static message
    while thread.is_alive():
        yield "🔎 AI Agent is researching... (Please wait)", ""
        time.sleep(0.5)
        
    # --- FINISHED ---
    sys.stdout = original_stdout  # Restore terminal
    
    # Combine the Report + The Source Stats (read from file; works across subprocesses)
    final_output = result_container["data"]
    from source_tracker import get_events
    source_stats = tracker.get_summary(direct_events=get_events())
    
    full_report = f"{final_output}\n\n---\n{source_stats}"
    
    yield "✅ Research Complete!", full_report


def flush_db():
    """Clear all papers from ChromaDB memory."""
    try:
        n = flush_memory()
        return f"✅ Flushed {n} papers from memory."
    except Exception as e:
        return f"❌ Error: {e}"


# --- 3. THE MINIMAL UI ---
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🧬 AI Research Agent")
    
    with gr.Row():
        msg = gr.Textbox(label="Research Topic", placeholder="e.g. VLA Manufacturing 2026")
        btn = gr.Button("🚀 Run", variant="primary", scale=0)
        flush_btn = gr.Button("🗑️ Flush Memory", variant="secondary")
    
    with gr.Row():
        # Simple Status Box
        status = gr.Label(label="Current Status", value="Ready")
    
    with gr.Row():
        # Final Report Area
        output = gr.Markdown(label="Final Research Report")

    # Connect buttons
    btn.click(fn=run_research, inputs=msg, outputs=[status, output])
    flush_btn.click(fn=flush_db, inputs=None, outputs=status)

if __name__ == "__main__":
    app.launch()