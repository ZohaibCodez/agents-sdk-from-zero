"""
Fantasy World Generator
----------------------
An interactive agent demo that builds a fantasy world using modular tools for magic, maps, and lore.
"""

import asyncio
import os
from dataclasses import dataclass
from typing import Any
import random

from dotenv import load_dotenv, find_dotenv
from openai import AsyncOpenAI
from agents import (
    Agent,
    Runner,
    OpenAIChatCompletionsModel,
    function_tool,
    set_tracing_disabled,
)
from agents.agent import ToolsToFinalOutputResult
from agents.model_settings import ModelSettings
from agents.run_context import RunContextWrapper
from agents.run import RunResultStreaming, ItemHelpers

# =========================
# Environment & Model Setup
# =========================

load_dotenv(find_dotenv())
set_tracing_disabled(True)
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    print("GEMINI_API_KEY not found.")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    openai_client=external_client, model="gemini-2.0-flash"
)

# =========================
# Tool Agent Definitions
# =========================

magic_agent: Agent = Agent(
    name="Magic System Agent",
    instructions="Describe an original and interesting magic system for a fantasy world.",
    model=model,
)

map_agent: Agent = Agent(
    name="World Map Agent",
    instructions="Generate a fantasy world map description including continents, cities, and terrain types.",
    model=model,
)

lore_agent: Agent = Agent(
    name="Lore Agent",
    instructions="Create ancient myths and legends for a fantasy world. Make them mysterious and powerful.",
    model=model,
)

# --- Convert Agents to Tools ---
magic_tool = magic_agent.as_tool(
    tool_name="generate_magic_system",
    tool_description="Generates a unique magic system for a fantasy world.",
)

map_tool = map_agent.as_tool(
    tool_name="generate_world_map",
    tool_description="Generates a fantasy world map description.",
)

lore_tool = lore_agent.as_tool(
    tool_name="generate_lore",
    tool_description="Creates ancient legends and lore for a fantasy setting.",
)

# =========================
# StoryMaster Agent
# =========================


@function_tool
async def generate_random_creature():
    """Returns a random magical creature name."""
    creatures: list[str] = [
        "Crystalwing Phoenix",
        "Shadowmane Unicorn",
        "Blazing Salamander",
        "Frostfang Dragon",
        "Void Panther",
    ]
    return random.choice(creatures)


story_master: Agent = Agent(
    name="StoryMasterAgent",
    instructions="""
    You are an agent - please keep going until the user's query is completely resolved, before ending your turn and yielding back to the user. Only terminate your turn when you are sure that the problem is solved. If you are not sure about file content or codebase structure pertaining to the user's request, use your tools to read files and gather the relevant information: do NOT guess or make up an answer.
    You MUST plan extensively before each function call, and reflect extensively on the outcomes of the previous function calls. DO NOT do this entire process by making function calls only, as this can impair your ability to solve the problem and think insightfully.

    You are a fantasy world builder assistant. When asked to build a world,
        use tools to:
        1. Generate a magic system
        2. Generate a world map
        3. Create ancient lore
        4. Add a magical creature
        Present everything as a fantasy setting.
    """,
    tools=[magic_tool, map_tool, lore_tool, generate_random_creature],
    model=model,
)


def print_section_header(title: str):
    """Prints a formatted section header."""
    print(f"\n{'='*60}\n{title}\n{'='*60}")


async def demo_agent_as_tool():
    """
    Demonstrates using agents as tools to build a fantasy world, showing tool calls and outputs as they happen.
    """
    print_section_header("DEMO: Fantasy World Generator (Agent as Tool)")
    try:
        result: RunResultStreaming = Runner.run_streamed(
            story_master,
            input="Create a fantasy world with a unique magic system, a detailed map, and some ancient myths. Also give me a random magical creature.",
        )
        print("=== Run starting ===")
        async for event in result.stream_events():
            # We'll ignore the raw responses event deltas
            if event.type == "raw_response_event":
                continue
            elif event.type == "agent_updated_stream_event":
                print(f"Agent updated: {event.new_agent.name}")
                continue
            elif event.type == "run_item_stream_event":
                if event.item.type == "tool_call_item":
                    print(f"-- Tool was called")
                elif event.item.type == "tool_call_output_item":
                    print(f"-- Tool output: {event.item.output}")
                elif event.item.type == "message_output_item":
                    print(
                        f"-- Message output:\n {ItemHelpers.text_message_output(event.item)}"
                    )
                else:
                    pass  # Ignore other event types
        print("=== Run complete ===")
        print("Final_Output : ", result.final_output)
    except Exception as e:
        print("  [ERROR] Fantasy World Generator:", e)


if __name__ == "__main__":
    asyncio.run(demo_agent_as_tool())
