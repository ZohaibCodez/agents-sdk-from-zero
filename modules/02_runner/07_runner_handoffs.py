"""
07_runner_handoffs.py

Demonstrates how the Runner handles agent handoffs in a custom smart assistant scenario.

Scenario:
- A `SmartAssistantRouter` analyzes user queries and routes to either:
  - `StudyAgent` for academic help
  - `WellbeingAgent` for emotional or productivity issues

Concepts:
- Runner reruns the loop after agent handoff
- Router agent uses handoff descriptions to decide
- Follows agent loop and handoff design from OpenAI SDK

Inspired by: https://openai.github.io/openai-agents-python/running_agents/#the-agent-loop

Environment variables:
    - OPENROUTER_API_KEY: Required for OpenRouter API access.

Author: Zohaib Khan
"""

import asyncio
import os
from dotenv import load_dotenv, find_dotenv
from openai import AsyncOpenAI
from agents import (
    Agent,
    Runner,
    OpenAIChatCompletionsModel,
    set_tracing_disabled,
)
from pprint import pprint

# =========================
# Environment & Model Setup
# =========================

# Load environment variables
load_dotenv(find_dotenv())
set_tracing_disabled(True)

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    print("OPENROUTER_API_KEY not found.")

external_client = AsyncOpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
)

model = OpenAIChatCompletionsModel(
    openai_client=external_client, model="deepseek/deepseek-chat-v3-0324:free"
)

# =========================
# Specialized Agents
# =========================

study_agent = Agent(
    name="StudyAgent",
    instructions="You are a study assistant. Help users with study tips, concentration issues, and academic planning.",
    model=model,
    handoff_description="For questions about studying, learning strategies, or academic productivity.",
)

wellbeing_agent = Agent(
    name="WellbeingAgent",
    instructions="You are a wellbeing assistant. Help users manage stress, motivation, mood, and productivity challenges.",
    model=model,
    handoff_description="For emotional support, motivation, mood, or productivity advice.",
)

# =========================
# Router Agent
# =========================

router_agent = Agent(
    name="SmartAssistantRouter",
    instructions=(
        "You are a smart assistant router. Decide whether the user's query is about study or wellbeing, "
        "and handoff to the correct agent. Use the agent's handoff_description in your reasoning if you hand off."
        "\n\nAvailable agents:"
        f"\n- {study_agent.name}: {study_agent.handoff_description}"
        f"\n- {wellbeing_agent.name}: {wellbeing_agent.handoff_description}"
    ),
    model=model,
    handoffs=[study_agent, wellbeing_agent],
)

# =========================
# Demo Function
# =========================

async def demo_runner_facilitating_handoff(user_query: str, expected_handler_name: str):
    """
    Run the router agent with a user query and print the final output and last agent.
    Args:
        user_query (str): The user's query to route.
        expected_handler_name (str): The expected agent to handle the query.
    """
    print("\n" + "="*70)
    print(f"Demo: Runner facilitating handoff for query: '{user_query}'")
    print("="*70)

    result = await Runner.run(starting_agent=router_agent, input=user_query)

    print(f"Final Output: {result.final_output}")
    print(f"Handled by Agent: {result.last_agent.name}")
    print(f"Expected Handler: {expected_handler_name}")
    # print(f"History:\n{result.to_input_list()}")

# =========================
# Main Test Cases
# =========================

async def run_handoff_demo_suite():
    """
    Main entry point for the handoff demo suite. Runs several test cases to demonstrate agent handoff.
    """
    print("\n--- Running 07_runner_handoffs.py (custom demo) ---")

    await demo_runner_facilitating_handoff(
        user_query="I keep procrastinating while studying for exams. What should I do?",
        expected_handler_name=wellbeing_agent.name,
    )

    await demo_runner_facilitating_handoff(
        user_query="How can I make a realistic study schedule for finals?",
        expected_handler_name=study_agent.name,
    )

    await demo_runner_facilitating_handoff(
        user_query="What's the latest AI news today?",
        expected_handler_name=router_agent.name,  # Should either handle or ask follow-up
    )

    print("\n--- Finished demo ---\n")

if __name__ == "__main__":
    try:
        asyncio.run(run_handoff_demo_suite())
    except Exception as e:
        print(f"\n❌ Exception occurred: {e}")
        print(f"Error type: {type(e).__name__}")
