"""
05_streaming_items_demo.py

🎭 Streaming Items Demonstration – "Joker" Agent with Tools

This module demonstrates how to use the OpenAI Agents SDK's streaming runner
to execute an agent that uses tools and streams different types of events
using `Runner.run_streamed()`.

🧠 Concept Demonstrated:
- Running agents with streaming support and tools
- Handling different event types (raw_response, agent_updated, run_item)
- Processing tool calls and outputs in real-time
- Using ItemHelpers for message output formatting

🤖 Agent Role:
A humorous "Joker" who uses tools to determine how many jokes to tell.

Author: Zohaib Khan
"""

import os
import asyncio
import random
from typing import Optional, List, Dict, Any
from pprint import pprint
from dotenv import load_dotenv, find_dotenv

from agents import (
    Agent,
    Runner,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    set_tracing_disabled,
    function_tool,
    RunResultStreaming,
    ItemHelpers
)

# =========================
# Environment & Model Setup
# =========================

# Load environment variables and disable tracing for cleaner output
load_dotenv(find_dotenv())
set_tracing_disabled(True)

# Configuration constants
GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
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


@function_tool
def how_many_jokes() -> int:
    """
    Tool that returns a random number of jokes to tell.
    
    Returns:
        int: Random number between 1 and 10
    """
    return random.randint(1, 10)


async def run_joker_demo() -> None:
    """
    Main runner function that executes the Joker agent with tools.
    
    This function demonstrates:
    1. Creating a joker agent with tool capabilities
    2. Streaming the agent's response with real-time event processing
    3. Handling different types of streaming events (raw_response, agent_updated, run_item)
    4. Processing tool calls and outputs as they occur
    """
    print("=" * 60)
    print("🎪 Starting Joker Agent with Tools Demo")
    print("=" * 60)
    joker: Agent = Agent(
        name="Joker",
        instructions="Act as a joker. First call the `how_many_jokes` tool, then tell that many jokes.",
        tools=[how_many_jokes],
        model=model,
    )
    print("✅ Joker agent is ready.\n")

    # First user query
    user_input_1: str = "Hello Joker"
    print(f"🎯 User input: '{user_input_1}'\n")
    print("📡 Streaming Response (Turn 1):\n" + "-" * 60)

    result: RunResultStreaming = Runner.run_streamed(joker, user_input_1)
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
                print("-- Tool was called")
            elif event.item.type == "tool_call_output_item":
                print(f"-- Tool output: {event.item.output}")
            elif event.item.type == "message_output_item":
                print(
                    f"-- Message output:\n {ItemHelpers.text_message_output(event.item)}"
                )
            else:
                pass  # Ignore other event types

    print("=== Run complete ===")

    print("\n✅ Joker demo completed successfully!\n")


if __name__ == "__main__":
    try:
        asyncio.run(run_joker_demo())
    except Exception as e:
        print(f"\n❌ Exception occurred: {e}")
