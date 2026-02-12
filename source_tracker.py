"""
Shared event tracker for tool usage. Uses a file so it works even when
CrewAI runs tools in a subprocess (main process can still read events).
"""
import json
import logging
import os

logger = logging.getLogger(__name__)

# File in project dir; works across processes
EVENTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".source_events.jsonl")


def clear_events():
    """Clear events at start of each research run."""
    try:
        if os.path.exists(EVENTS_FILE):
            os.remove(EVENTS_FILE)
    except Exception as e:
        logger.warning(f"Could not clear events file: {e}")


def append_event(event):
    """Append an event. Works from any process/subprocess."""
    try:
        with open(EVENTS_FILE, "a", encoding="utf-8") as f:
            line = json.dumps(event, ensure_ascii=False) + "\n"
            f.write(line)
    except Exception as e:
        logger.warning(f"Could not append event: {e}")


def get_events():
    """Read all events. Returns list of (str | [type, payload])."""
    events = []
    try:
        if os.path.exists(EVENTS_FILE):
            with open(EVENTS_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except Exception as e:
        logger.warning(f"Could not read events file: {e}")
    return events
