import gradio as gr
import sys
import threading
import time
import os
from crew import run_crew
from memory import flush_memory
# --- NEW IMPORT ---
from validation import validate_report

# --- CONFIG & ASSETS ---
IMG_IDLE = os.path.join("videos", "hello.gif")
IMG_SEARCH = os.path.join("videos", "research.gif")
IMG_DONE = os.path.join("videos", "output.gif")

# --- 1. THE HIDDEN LOGGER ---
class SourceTracker:
    def __init__(self):
        self.logs = []
        
    def write(self, text):
        clean = text.strip()
        if clean: self.logs.append(clean)

    def flush(self): pass

    def get_summary(self, direct_events=None):
        mem_hits = sum(1 for line in self.logs if "🧠" in line)
        web_hits = sum(1 for line in self.logs if "🌐" in line)
        
        if direct_events:
            mem_hits = sum(1 for e in direct_events if e == "mem")
            web_hits = sum(1 for e in direct_events if e == "web")
            
        return f"""
        ### 📊 Data Source Summary
        - **Local Memory (ChromaDB):** Used {mem_hits} times (Fast & Free)
        - **External API (OpenAlex):** Used {web_hits} times (Slow & Costly)
        """

# --- 2. THE RESEARCH FUNCTION ---
def run_research(topic):
    from source_tracker import clear_events, get_events
    clear_events()
    
    # Setup tracking
    tracker = SourceTracker()
    original_stdout = sys.stdout
    sys.stdout = tracker
    
    # Run AI in background thread
    result_container = {"data": None}
    def target():
        try:
            result_container["data"] = str(run_crew(topic))
        except Exception as e:
            result_container["data"] = f"Error: {e}"
    
    thread = threading.Thread(target=target)
    thread.start()
    
    # --- UI UPDATE LOOP ---
    while thread.is_alive():
        yield "🔎 Dr. Kimi is reading papers... (Please wait)", "", IMG_SEARCH
        time.sleep(0.5)
        
    # --- FINISHED ---
    sys.stdout = original_stdout
    
    final_output = result_container["data"]
    
    # --- NEW: RUN VALIDATION ---
    # We check the file that 'crew.py' just created
    val_report = ""
    try:
        # Check if file exists before validating
        if os.path.exists("final_research_report.md"):
            v_result = validate_report("final_research_report.md")
            
            val_report = "\n\n---\n### 🛡️ Quality Assurance Check\n"
            if v_result["validation_passed"]:
                val_report += f"✅ **PASSED**: No hallucinations detected.\n"
                val_report += f"- **Valid Citations:** {v_result['citation_stats']['valid_links']}/{v_result['citation_stats']['total_citations']}\n"
            else:
                val_report += f"⚠️ **ISSUES FOUND**:\n"
                for issue in v_result["issues"]:
                    val_report += f"- ❌ {issue}\n"
        else:
            val_report = "\n\n---\n⚠️ **Validation Skipped**: Output file not found."
            
    except Exception as e:
        val_report = f"\n\n---\n⚠️ Validation Error: {str(e)}"

    # Generate Stats
    source_stats = tracker.get_summary(direct_events=get_events())
    
    # Combine: Report + Validation + Stats
    full_report = f"{final_output}{val_report}\n\n---\n{source_stats}"
    
    yield "✅ Research Complete!", full_report, IMG_DONE


def flush_db():
    try:
        n = flush_memory()
        return f"✅ Flushed {n} papers."
    except Exception as e:
        return f"❌ Error: {e}"

# --- 3. THE WORLD-CLASS UI ---
custom_css = """
#title_area {text-align: center; margin-bottom: 20px;}
#avatar_image {border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border: none;}
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="pink", secondary_hue="indigo"), css=custom_css) as app:
    
    with gr.Row(elem_id="title_area"):
        gr.Markdown('''
        # 🌸 Dr. Kimi 😺
        ### Your Personal AI Research Assistant 🌐
        #### using OpenAlex 🔎
        ''')

    with gr.Row():
        with gr.Column(scale=1):
            avatar = gr.Image(
                value=IMG_IDLE, 
                label="Agent State", 
                show_label=False, 
                elem_id="avatar_image",
                interactive=False,
                type="filepath",
                height=350
            )
            
        with gr.Column(scale=2):
            msg = gr.Textbox(
                label="Research Topic", 
                placeholder="e.g. VLA Manufacturing 2026",
                lines=2
            )
            
            with gr.Row():
                btn = gr.Button("🌻 Run Research", variant="primary", scale=2)
                flush_btn = gr.Button("🗑️ Flush Memory", variant="secondary", scale=1)
            
            status = gr.Label(value="Ready to help!", label="Current Status")

    with gr.Row():
        output = gr.Markdown(label="Final Research Report")

    btn.click(
        fn=run_research, 
        inputs=msg, 
        outputs=[status, output, avatar]
    )
    
    flush_btn.click(fn=flush_db, inputs=None, outputs=status)

if __name__ == "__main__":
    app.launch()