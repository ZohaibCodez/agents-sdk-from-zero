"""
03_run_result_advanced.py

Demonstrates advanced RunResult usage with tools, handoffs, and error scenarios using the OpenAI Agents SDK.

Features demonstrated:
- RunResult with tool usage and complex new_items
- Handoff scenarios and agent transitions
- Error handling and partial results
- Performance analysis and timing
- Memory usage and optimization patterns

Based on:
- https://openai.github.io/openai-agents-python/results/
- https://openai.github.io/openai-agents-python/tools/
- https://openai.github.io/openai-agents-python/handoffs/

Environment variables:
    - GEMINI_API_KEY: Required for Gemini model access.

Author: Zohaib Khan
"""

import asyncio
import os
from dotenv import load_dotenv, find_dotenv
from openai import AsyncOpenAI
from openai.types.responses import ResponseTextDeltaEvent
from agents import (
    Agent,
    Runner,
    OpenAIChatCompletionsModel,
    RunResult,
    handoff,
    set_tracing_disabled,
    function_tool,
)
from pprint import pprint
import random

# =========================
# Environment & Model Setup
# =========================

load_dotenv(find_dotenv())
set_tracing_disabled(True)

# Configuration constants
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")
GEMINI_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_MODEL_NAME: str = "gemini-2.0-flash"

# Validate API key
if not GEMINI_API_KEY:
    print("❌ GEMINI_API_KEY environment variable is required but not found.")
    raise ValueError("GEMINI_API_KEY environment variable is required but not found.")

print(f"🚀 Initializing Gemini client with model: {GEMINI_MODEL_NAME}")

# Initialize external OpenAI client for Gemini
external_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url=GEMINI_BASE_URL,
)

# Configure the chat completions model
model = OpenAIChatCompletionsModel(
    openai_client=external_client, model=GEMINI_MODEL_NAME
)

print(f"✅ Model configured successfully\n")

# =========================
# Tool Functions
# =========================


@function_tool
def calculate(a: int, b: int) -> int:
    """Tool to add two numbers."""
    return a + b


@function_tool
def get_random_number() -> int:
    """Tool to get a random integer between 1 and 10."""
    return random.randint(1, 10)


# =========================
# Specialized Agents
# =========================

math_agent: Agent = Agent(
    name="Math Agent",
    instructions="Act as a mathematician. Solve numeric problems using the calculate tool.",
    model=model,
    tools=[calculate],
)

joke_agent: Agent = Agent(
    name="Joke Agent",
    instructions="Act as a joker. Use get_random_number to get a number. And for that number tell that many jokes.",
    model=model,
    tools=[get_random_number],
)

coordinator_agent: Agent = Agent(
    name="Coordinator Agent",
    instructions="""Act as a coordinator agent.
    - For Math Queries, handoff to math_agent
    - For Joke Queries, handoff to joke_agent
    - For other queries, handle them yourself
""",
    model=model,
    handoffs=[
        handoff(math_agent, tool_description_override="Solve math problems"),
        handoff(
            joke_agent, tool_description_override="Tell jokes using a random number"
        ),
    ],
)

# =========================
# RunResult Analysis Function
# =========================


async def analyze_run_result(run_result: RunResult, scenario_name: str):
    """
    Prints a detailed analysis of a RunResult for a given scenario.
    Shows final output, last agent, input, new items, and conversation history.
    """
    print("\n==============================")
    print(f"📊 Analysis for: {scenario_name}")
    print("==============================")
    print(f"🧠 Final Output: {run_result.final_output}")
    print(f"🤖 Last Agent: {run_result.last_agent.name}")
    print(f"📨 Original Input: {run_result.input}")

    # run_result.new_items
    print(f"\n🧾 New Items (Count: {len(run_result.new_items)}):")
    for i, item in enumerate(run_result.new_items, start=1):
        print(f"  {i}. {type(item).__name__} → {str(item)[:60]}...")

    print(f"\n💬 Conversation History (Length: {len(run_result.to_input_list())}):")
    for i, item in enumerate(run_result.to_input_list(), start=1):
        print(f"   - Item {i}: ")
        pprint(item)
        print()


# =========================
# Main Demo Function
# =========================


async def run_result_advanced_demo():
    """
    Runs several advanced RunResult scenarios with tools, handoffs, and agent transitions.
    Prints detailed analysis for each scenario.
    """
    scenarios: list[tuple[str, str]] = [
        ("Simple Math", "What is 5 + 3?"),
        ("Joke Request", "Tell me a joke."),
        ("Mixed Task", "Add 10 + 20 and also tell me a joke."),
        ("General Question", "What is the capital of Japan?"),
    ]
    for title, user_input in scenarios:
        result = await Runner.run(starting_agent=coordinator_agent, input=user_input)
        await analyze_run_result(result, title)


# =========================
# Main Entry Point
# =========================

if __name__ == "__main__":
    try:
        asyncio.run(run_result_advanced_demo())
    except Exception as e:
        print(f"\n❌ Exception occurred: {e}")
        print(f"Error type: {type(e).__name__}")
