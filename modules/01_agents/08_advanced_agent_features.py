"""
Module: advanced_agent_features.py

Demonstrates advanced features of the OpenAI Agents SDK, including agent cloning, agent-as-tool, and tool-use behavior control.

- Shows how to clone agents with different personalities.
- Demonstrates using agents as callable tools.
- Illustrates controlling tool execution flow with tool_use_behavior.

Environment variables:
    - GEMINI_API_KEY: Required for Gemini model access.

Author: [Your Name or Organization]
"""

import asyncio
import os
import random
from dataclasses import dataclass, field
from typing import Any

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

# Load environment variables from .env file
load_dotenv(find_dotenv())
set_tracing_disabled(True)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("GEMINI_API_KEY not found.")

external_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    openai_client=external_client, model="gemini-2.0-flash"
)

# =====================================
# DEMO 1: Agent Cloning (with Variations)
# =====================================


def _print_section_header(title: str):
    """Print a formatted section header for console output."""
    print(f"\n{'='*60}\n{title}\n{'='*60}")


async def demo_agent_cloning():
    """
    Demonstrates cloning an agent with different personalities and instructions.
    Each clone has a unique translation style.
    """
    _print_section_header("Demo 1: Agent Cloning - Setting up base agent and clones...")

    base_translator_agent: Agent = Agent(
        name="Base Translator Agent",
        instructions="You are a language translator agent. Translate the given sentence clearly.",
        model=model,
        model_settings=ModelSettings(temperature=0.9),
    )

    # Clone 1: Formal Urdu (Roman script)
    formal_urdu_agent: Agent = base_translator_agent.clone(
        instructions="Translate the sentence into formal Urdu using respectful and grammatically correct tone. Use Roman Urdu script (not Urdu letters).",
        model_settings=ModelSettings(temperature=0.4),
    )

    # Clone 2: Emojifier
    emojifier_agent: Agent = base_translator_agent.clone(
        instructions="Translate the meaning of the sentence into expressive emojis. Don't write any words unless needed for clarity.",
        model_settings=ModelSettings(temperature=0.9),
    )

    # Clone 3: French Poet
    french_poet_agent: Agent = base_translator_agent.clone(
        instructions="Translate the sentence into poetic French, as if written by a romantic French poet. Add a touch of metaphor or emotion.",
        model_settings=ModelSettings(temperature=0.7),
    )

    # Clone 4: Shakespearean
    shakespeare_agent: Agent = base_translator_agent.clone(
        instructions="Translate the sentence as if spoken by Shakespeare. Use Early Modern English and poetic style.",
        model_settings=ModelSettings(temperature=0.8),
    )

    query = "I love programming because it feels like solving a puzzle."

    print("\n[STEP] Base Translator Agent:")
    try:
        result = await Runner.run(base_translator_agent, query)
        print("  [RESULT] Base Translator:", result.final_output)
    except Exception as e:
        print("  [ERROR] Base Translator:", e)

    print("\n[STEP] Formal Urdu Agent:")
    try:
        result = await Runner.run(formal_urdu_agent, query)
        print("  [RESULT] Formal Urdu:", result.final_output)
    except Exception as e:
        print("  [ERROR] Formal Urdu:", e)

    print("\n[STEP] Emojifier Agent:")
    try:
        result = await Runner.run(emojifier_agent, query)
        print("  [RESULT] Emojifier:", result.final_output)
    except Exception as e:
        print("  [ERROR] Emojifier:", e)

    print("\n[STEP] French Poet Agent:")
    try:
        result = await Runner.run(french_poet_agent, query)
        print("  [RESULT] French Poet:", result.final_output)
    except Exception as e:
        print("  [ERROR] French Poet:", e)

    print("\n[STEP] Shakespearean Agent:")
    try:
        result = await Runner.run(shakespeare_agent, query)
        print("  [RESULT] Shakespearean:", result.final_output)
    except Exception as e:
        print("  [ERROR] Shakespearean:", e)


# =========================
# Agent.as_tool() Example
# =========================


@function_tool
def generate_random_creature() -> str:
    """Return a random magical creature name."""
    creatures: list[str] = [
        "Crystalwing Phoenix",
        "Shadowmane Unicorn",
        "Blazing Salamander",
        "Frostfang Dragon",
        "Void Panther",
    ]
    return random.choice(creatures)


# --- Tool Agents ---
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

# --- StoryMaster Agent ---
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


async def demo_agent_as_tool():
    """
    Demonstrates using agents as tools to build a fantasy world, showing tool calls and outputs as they happen.
    """
    _print_section_header("DEMO 2: Fantasy World Generator (Agent as Tool)")
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


# =============================================================================
# tool_use_behavior - Controlling Tool Execution Flow
# =============================================================================


@dataclass
class TaskContext:
    task_count: int = 0
    completed_tasks: list[str] | None = None

    def __post_init__(self):
        if self.completed_tasks is None:
            self.completed_tasks = []


@function_tool
async def calculate_sum(context: RunContextWrapper, numbers: list[int]) -> str:
    """Calculate the sum of a list of numbers"""
    print("📅 [DEBUG] Called: calculate_sum")
    total = sum(numbers)
    context.context.task_count += 1
    return f"Sum of {numbers} = {total}"


@function_tool
async def calculate_product(context: RunContextWrapper, numbers: list[int]) -> str:
    """Calculate the product of a list of numbers"""
    print("📅 [DEBUG] Called: calculate_product")
    product = 1
    for num in numbers:
        product *= num
    context.context.task_count += 1
    return f"Product of {numbers} = {product}"


@function_tool
async def final_calculation(
    context: RunContextWrapper, operation: str, result: str
) -> str:
    """Mark a calculation as complete"""
    print("📅 [DEBUG] Called: final_calculation")
    context.context.completed_tasks.append(f"{operation}: {result}")
    return f"✅ Completed {operation} - {result}"


async def demo_tool_use_behavior():
    """Demonstrate different tool_use_behavior settings"""
    print("=" * 60)
    print("DEMO 3: Tool Use Behavior")
    print("=" * 60)

    context = TaskContext()

    # 1. Default behavior: "run_llm_again"
    print("🔄 Default Behavior (run_llm_again):")
    agent_default = Agent(
        name="DefaultAgent",
        instructions="You are a calculator. Use tools to perform calculations and explain the results.",
        tools=[calculate_sum, calculate_product],
        tool_use_behavior="run_llm_again",  # Default
        model=model,
    )

    result = await Runner.run(
        agent_default, "Calculate the sum of [1, 2, 3, 4, 5]", context=context
    )
    print(f"   {result.final_output}")
    print(f"   Task count: {context.task_count}\n")

    # 2. Stop on first tool
    print("⏹️  Stop on First Tool:")
    context = TaskContext()  # Reset context
    agent_stop_first = Agent(
        name="StopFirstAgent",
        instructions="Use tools to perform calculations.",
        tools=[calculate_sum, calculate_product],
        tool_use_behavior="stop_on_first_tool",
        model=model,
    )

    result = await Runner.run(
        agent_stop_first,
        "Calculate the sum of [1, 2, 3, 4, 5] and mark it as complete",
        context=context,
    )
    print(f"   {result.final_output}")
    print(f"   Task count: {context.task_count}\n")

    # 3. Stop at specific tools
    print("🎯 Stop at Specific Tools:")
    context = TaskContext()  # Reset context
    agent_stop_specific = Agent(
        name="StopSpecificAgent",
        instructions="Use tools to perform calculations.",
        tools=[calculate_sum, calculate_product, final_calculation],
        tool_use_behavior={"stop_at_tool_names": ["final_calculation"]},
        model=model,
    )

    result = await Runner.run(
        agent_stop_specific,
        "Calculate the sum of [1, 2, 3] and mark it as complete",
        context=context,
    )
    print(f"   {result.final_output}")
    print(f"   Task count: {context.task_count}\n")
    print(f"   Completed tasks: {context.completed_tasks}\n")


if __name__ == "__main__":
    # asyncio.run(demo_agent_cloning())
    # asyncio.run(demo_agent_as_tool())
    asyncio.run(demo_tool_use_behavior())
