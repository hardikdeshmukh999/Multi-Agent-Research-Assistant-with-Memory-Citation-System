# app.py
import streamlit as st
from crew import run_crew
from st_callback import StreamlitCallbackHandler
import sys

# Redirect system output to catch some lower-level logs
class StreamToStreamlit:
    def __init__(self, container):
        self.container = container
    def write(self, data):
        # Filter out empty lines to look cleaner
        if data.strip():
            self.container.markdown(f"_{data.strip()}_")
    def flush(self):
        pass

st.set_page_config(page_title="AI Researcher 2026", layout="wide")

st.title("🧬 AI Research Collective")
st.markdown("Automated academic synthesis using Multi-Agent Systems.")

topic = st.text_input("Enter Research Topic:", placeholder="e.g., Genetic engineering in 2026")

if st.button("🚀 Start Research"):
    if topic:
        # The "Live" Status Container
        with st.status("🕵️ Agents are investigating (this may take 1-2 mins)...", expanded=True) as status:
            
            # 1. Create a container for the live text
            log_container = st.empty()
            
            # 2. Redirect standard output (print statements) to that container
            # This captures "Searching OpenAlex..." logs automatically
            sys.stdout = StreamToStreamlit(log_container)
            
            # 3. Run the Crew
            try:
                result = run_crew(topic)
                status.update(label="✅ Research Complete!", state="complete", expanded=False)
            except Exception as e:
                st.error(f"An error occurred: {e}")
                result = ""

        # Reset stdout so other things don't print to the UI weirdly
        sys.stdout = sys.__stdout__

        if result:
            st.subheader("Final Research Report")
            st.markdown(result)
            
            st.download_button(
                label="📥 Download Report",
                data=str(result),
                file_name=f"{topic.replace(' ', '_')}_report.md",
                mime="text/markdown"
            )
    else:
        st.error("Please enter a topic!")