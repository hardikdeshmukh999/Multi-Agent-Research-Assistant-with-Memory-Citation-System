# crew.py
import os
import yaml
from crewai import Agent, Task, Crew, Process, LLM
from tools import search_openalex
from dotenv import load_dotenv

load_dotenv()

# Helper to load YAML
def load_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

agents_config = load_config('agents.yaml')
tasks_config = load_config('tasks.yaml')

# --- DEFINING THE OPENAI MODELS ---
# 1. The "Budget" Brain (Librarian & Critic) - gpt-4o-mini
mini_llm = LLM(model="openai/gpt-4o-mini")

# 2. The "Genius" Brain (Scribe) - gpt-4o
smart_llm = LLM(model="openai/gpt-4o")

# --- AGENTS ---
librarian = Agent(
    config=agents_config['librarian'],
    tools=[search_openalex],
    llm=mini_llm,
    verbose=True,
    allow_delegation=False
)

critic = Agent(
    config=agents_config['critic'],
    llm=mini_llm,
    verbose=True,
    allow_delegation=False
)

scribe = Agent(
    config=agents_config['scribe'],
    llm=smart_llm,
    verbose=True,
    allow_delegation=False
)

# --- TASKS ---
research_task = Task(
    config=tasks_config['research_task'],
    agent=librarian
)

review_task = Task(
    config=tasks_config['review_task'],
    agent=critic,
    context=[research_task]
)

synthesis_task = Task(
    config=tasks_config['synthesis_task'],
    agent=scribe,
    context=[review_task],
    output_file='final_research_report.md'
)

# --- CREW ---
# Change from Process.sequential to Process.hierarchical
research_crew = Crew(
    agents=[librarian, critic, scribe],
    tasks=[research_task, review_task, synthesis_task],
    process=Process.hierarchical, # <--- The agent now manages its own hops
    manager_llm=smart_llm,        # GPT-4o acts as the "Project Manager"
    verbose=True
)
def run_crew(topic):
    print(f"🚀 Starting the '{topic}' Research Crew...")
    result = research_crew.kickoff(inputs={'topic': topic})
    return result

if __name__ == "__main__":
    # Test run
    run_crew("Future of AI Agents 2026")