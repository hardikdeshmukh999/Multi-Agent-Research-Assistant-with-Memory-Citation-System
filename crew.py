# crew.py
import os
import yaml
from crewai import Agent, Task, Crew, Process, LLM
from tools import search_openalex
from dotenv import load_dotenv

load_dotenv()

# --- LOAD CONFIGS ---
def load_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

agents_config = load_config('agents.yaml')
tasks_config = load_config('tasks.yaml')

# --- LLM DEFINITIONS ---
# Librarian & Critic: Fast & Cheap
mini_llm = LLM(model="openai/gpt-4o-mini")

# Scribe: Smart & Professional
smart_llm = LLM(model="openai/gpt-4o")

# --- AGENTS ---
librarian = Agent(
    config=agents_config['librarian'],
    tools=[search_openalex],  # Librarian HAS the tool
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
# We use 'context' to pass data from one task to the next
research_task = Task(
    config=tasks_config['research_task'],
    agent=librarian
)

review_task = Task(
    config=tasks_config['review_task'],
    agent=critic,
    context=[research_task]  # Critic reviews Librarian's work
)

synthesis_task = Task(
    config=tasks_config['synthesis_task'],
    agent=scribe,
    context=[review_task],  # Scribe writes based on Critic's review
    output_file='final_research_report.md'
)

# --- CREW (Sequential) ---
research_crew = Crew(
    agents=[librarian, critic, scribe],
    tasks=[research_task, review_task, synthesis_task],
    process=Process.sequential,  # Sequential execution
    verbose=True
)

# --- MAIN FUNCTION ---
def run_crew(topic):
    """
    Runs the research crew for a given topic.
    Returns the final report as a string.
    """
    print(f"🚀 Starting Sequential Research on: {topic}")
    
    try:
        # Kickoff the crew with the topic
        result = research_crew.kickoff(inputs={'topic': topic})
        
        # CrewAI returns a CrewOutput object, convert to string
        final_output = str(result)
        
        print("✅ Research completed successfully!")
        return final_output
        
    except Exception as e:
        error_msg = f"❌ Error during research: {str(e)}"
        print(error_msg)
        return error_msg

# --- TESTING ---
if __name__ == "__main__":
    # Only runs when you execute this file directly
    test_topic = "AI Agents in Software Engineering"
    result = run_crew(test_topic)
    print("\n" + "="*50)
    print("FINAL RESULT:")
    print("="*50)
    print(result)