"""
Module: tool_behaviour.py

Demonstrates different tool use behaviors in the OpenAI Agents SDK, showing how to control
when and how agents use tools during execution.

Features demonstrated:
- Default behavior (run_llm_again)
- Stop on first tool (stop_on_first_tool)
- Stop at specific tools (StopAtTools)
- Custom tool behavior functions

Environment variables:
    - GEMINI_API_KEY: Required for Gemini model access.

Author: Zohaib Khan
"""

import asyncio
import os
import re
from dataclasses import dataclass, field
from typing import Any, List, Dict

from dotenv import load_dotenv, find_dotenv
from openai import AsyncOpenAI
from agents import (
    Agent,
    Runner,
    OpenAIChatCompletionsModel,
    function_tool,
    set_tracing_disabled,
)
from agents.agent import ToolsToFinalOutputResult, StopAtTools
from agents.run_context import RunContextWrapper

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

# =============================================================================
# Context and Tools
# =============================================================================

@dataclass
class ToolBehaviorContext:
    """Context for tracking tool execution behavior and results."""
    execution_log: list[str] = field(default_factory=list)
    tool_results: dict[str, Any] = field(default_factory=dict)
    processing_mode: str = "default"


@function_tool
async def get_dirty_data(ctx: RunContextWrapper[ToolBehaviorContext]) -> str:
    """Retrieve messy, uncleaned input data for processing."""
    data: str = "  heLlo!! ThIS   is A## MEsSy   dAtA @@strIng... "
    ctx.context.execution_log.append(f"TOOL: get_dirty_data called with data={data}")
    ctx.context.tool_results["get_dirty_data"] = data
    return data


@function_tool
async def clean_data(ctx: RunContextWrapper[ToolBehaviorContext], text: str) -> str:
    """Clean and normalize input text by removing special characters and formatting."""
    cleaned = re.sub(r"[^a-zA-Z0-9 ]", "", text)
    cleaned = " ".join(cleaned.split())
    cleaned = cleaned.capitalize()
    ctx.context.execution_log.append("TOOL: clean_data called")
    ctx.context.tool_results["clean_data"] = cleaned
    return cleaned

# =============================================================================
# Demo 1: Default Behavior (run_llm_again)
# =============================================================================

async def demo_default_behavior():
    """
    Demonstrate the default tool use behavior where the agent runs the LLM again
    after each tool call to decide the next action.
    """
    print("\n📚 DEMO 1 - run_llm_again: Data Cleaning Assistant")
    print("=" * 70)

    context = ToolBehaviorContext()

    agent = Agent(
        name="DataCleaningAgent",
        instructions="""
        You are a data cleaning assistant. 
        1. First get messy input text.
        2. Then clean the text using the tool.
        3. Finally, explain what was cleaned and provide the cleaned result.
        """,
        tools=[get_dirty_data, clean_data],
        # run_llm_again is the default behavior
        model=model,
    )

    query = "Clean the input text and explain the cleanup."

    result = await Runner.run(agent, query, context=context)

    print("🔄 Testing default behavior")
    print(f"[QUERY] {query}")
    print(f"\n✅ Final Output:\n{result.final_output}\n")

    print("📊 Execution Log:")
    for i, log in enumerate(context.execution_log, 1):
        print(f"  {i}. {log}")

    print("\n🔧 Tool Results:")
    for tool, output in context.tool_results.items():
        print(f"  {tool}: {output}")

# =============================================================================
# Demo 2: Stop on First Tool (stop_on_first_tool)
# =============================================================================

async def demo_stop_on_first_tool():
    """
    Demonstrate stopping execution after the first tool call, returning the tool result directly
    without further LLM processing.
    """
    print("\n📚 DEMO 2 - stop_on_first_tool: Data Cleaning Assistant")
    print("=" * 70)

    context = ToolBehaviorContext()

    agent = Agent(
        name="DataCleaningAgent",
        instructions="""
        You are a fast data fetcher.
        Use tools to retrieve unprocessed, messy input data.
        Do not clean or explain — return the first tool result directly.
        """,
        tools=[get_dirty_data, clean_data],
        tool_use_behavior="stop_on_first_tool",  # Stop after first tool call
        model=model,
    )

    query = "Get the raw uncleaned input data and cleaned it."

    result = await Runner.run(agent, query, context=context)

    print("⏹️  Testing stop_on_first_tool")
    print(f"[QUERY] {query}")
    print(f"\n✅ Final Output:\n{result.final_output}\n")

    print("📊 Execution Log:")
    for i, log in enumerate(context.execution_log, 1):
        print(f"  {i}. {log}")

    print("\n🔧 Tool Results:")
    for tool, output in context.tool_results.items():
        print(f"  {tool}: {output}")

# =============================================================================
# Demo 3: Stop at Specific Tools (StopAtTools)
# =============================================================================

async def demo_stop_at_specific_tools():
    """
    Demonstrate stopping execution when a specific tool is called,
    providing selective control over tool execution flow.
    """
    print("\n📚 DEMO 3 - StopAtTools: Data Cleaning Assistant")
    print("=" * 70)

    context = ToolBehaviorContext()

    agent = Agent(
        name="DataCleaningAgent",
        instructions="""
        You are a data cleaning agent.
        - First fetch messy text.
        - Then clean the text.
        - Do not explain the cleanup — just return cleaned output.
        Stop once the cleaning is done.
        """,
        tools=[get_dirty_data, clean_data],
        tool_use_behavior=StopAtTools(
            stop_at_tool_names=["clean_data"]
        ),  # Stop when clean_data is called
        model=model,
    )

    query = "Clean the messy input data and return only the cleaned version."

    result = await Runner.run(agent, query, context=context)

    print("🎯 Testing StopAtTools behavior")
    print(f"[QUERY] {query}")
    print(f"\n✅ Final Output:\n{result.final_output}\n")

    print("📊 Execution Log:")
    for i, log in enumerate(context.execution_log, 1):
        print(f"  {i}. {log}")

    print("\n🔧 Tool Results:")
    for tool, output in context.tool_results.items():
        print(f"  {tool}: {output}")

# =============================================================================
# Demo 4: Custom Tool Behavior Function
# =============================================================================

async def custom_tool_behavior(
    context: RunContextWrapper[ToolBehaviorContext], tool_results: List[Any]
) -> ToolsToFinalOutputResult:
    """
    Custom function to determine when to stop based on tool results and conditions.
    
    Args:
        context: The run context containing execution state
        tool_results: List of results from previous tool calls
        
    Returns:
        ToolsToFinalOutputResult indicating whether to stop or continue
    """
    context.context.execution_log.append(
        f"CUSTOM: Evaluating {len(tool_results)} tool results"
    )

    if not tool_results:
        return ToolsToFinalOutputResult(is_final_output=False)

    last_result = tool_results[-1]

    # Extract plain result from tool output
    result_text = (
        last_result.result if hasattr(last_result, "result") else str(last_result)
    )
    word_count = len(result_text.split())

    # Define custom stop conditions
    stop_conditions = [
        len(tool_results) >= 1,  # Stop after at least one tool call
        word_count < 10,         # Stop if result is very short
        "error" in result_text.lower(),  # Stop if error detected
    ]

    if any(stop_conditions):
        context.context.execution_log.append("CUSTOM: Stopping with custom output.")

        # Create custom summary output
        summary = f"🧹 CUSTOM CLEANING SUMMARY\nWords: {word_count}\nFinal cleaned result:\n{result_text[:100]}..."
        return ToolsToFinalOutputResult(is_final_output=True, final_output=summary)

    context.context.execution_log.append("CUSTOM: Continue execution.")
    return ToolsToFinalOutputResult(is_final_output=False)


async def demo_custom_tool_behavior():
    """
    Demonstrate custom tool behavior function that provides advanced control
    over when to stop tool execution based on custom logic.
    """
    print("\n📚 DEMO 4 - Custom Tool Behavior: Data Cleaning Assistant")
    print("=" * 70)

    context = ToolBehaviorContext()

    agent = Agent(
        name="CustomCleaningAgent",
        instructions="""
        You are an intelligent data cleaning agent.
        Fetch dirty input → clean it → system decides when to stop.
        Use tools as needed and let the system determine the end.
        """,
        tools=[get_dirty_data, clean_data],
        tool_use_behavior=custom_tool_behavior,  # Custom behavior function
        model=model,
    )

    query = "Clean messy data with smart logic."

    result = await Runner.run(agent, query, context=context)

    print("🧠 Testing custom tool behavior")
    print(f"[QUERY] {query}")
    print(f"\n✅ Final Output:\n{result.final_output}\n")

    print("📊 Execution Log:")
    for i, log in enumerate(context.execution_log, 1):
        print(f"  {i}. {log}")

    print("\n🔧 Tool Results:")
    for tool, output in context.tool_results.items():
        print(f"  {tool}: {output}")

# =============================================================================
# Main Demo Runner
# =============================================================================

async def run_all_tool_behavior_demos():
    """Run all tool behavior demonstration scenarios."""
    print("🚀 Starting Tool Behavior Demo Suite")
    print("=" * 60)
    
    # Uncomment the demo you want to run:
    # await demo_default_behavior()
    # await demo_stop_on_first_tool()
    # await demo_stop_at_specific_tools()
    await demo_custom_tool_behavior()


if __name__ == "__main__":
    asyncio.run(run_all_tool_behavior_demos())
