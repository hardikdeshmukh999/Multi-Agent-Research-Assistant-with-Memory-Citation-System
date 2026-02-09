# st_callback.py
import streamlit as st
from langchain_core.callbacks import BaseCallbackHandler

class StreamlitCallbackHandler(BaseCallbackHandler):
    """Callback handler to stream agent 'thoughts' to a Streamlit status container."""
    def __init__(self, status_container):
        self.status_container = status_container

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        pass

    def on_agent_action(self, action, **kwargs):
        self.status_container.write(f"🤖 **{action.tool}**: {action.tool_input}")