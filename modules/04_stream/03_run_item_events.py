"""
03_run_item_events.py

Demonstrates run item events and agent events for higher-level streaming using the OpenAI Agents SDK.

Features demonstrated:
- Stream higher-level events like tool calls, messages, and agent updates
- Analyze event types and item types
- Print event progress in a user-friendly format

Based on:
- https://openai.github.io/openai-agents-python/streaming/
- Run item events and agent events section

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
    RunResultStreaming,
    set_tracing_disabled,
    function_tool,
    ItemHelpers
)
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
def get_random_number() -> int:
    """Tool to get a random integer between 1 and 10."""
    return random.randint(1, 10)

# =========================
# Specialized Agents
# =========================

joke_agent: Agent = Agent(
    name="Joke Agent",
    instructions="Act as a joker. Use get_random_number to get a number. And for that number tell that many jokes.",
    model=model,
    tools=[get_random_number],
)

# =========================
# Streaming Demo Functions
# =========================

async def stream_hight_level_events():
    """
    Streams higher-level agent events (tool calls, messages, agent updates) and prints them in a user-friendly format.
    """
    user_input: str = "Tell me some jokes"
    print("\n==============================")
    print(" 03_run_item_events.py Demo: High-Level Events")
    print("==============================\n")
    print(f"[User Input]\n{user_input}\n")

    run_result_stream: RunResultStreaming = Runner.run_streamed(
        starting_agent=joke_agent, input=user_input
    )

    print("------------------------------")
    print("[Streaming High-Level Events]")
    print("------------------------------\n")
    async for event in run_result_stream.stream_events():
        if event.type == "raw_response_event":
            continue
        elif event.type == "agent_updated_stream_event":
            print(f"🤖 Agent updated: {event.new_agent.name}")
        elif event.type == "run_item_stream_event":
            item = event.item
            if item.type == "tool_call_item":
                print("🔧 Tool called")
            elif item.type == "tool_call_output_item":
                print(f"📤 Tool output: {item.output}")
            elif item.type == "message_output_item":
                print(f"💬 Message output:\n{ItemHelpers.text_message_output(event.item)}")
            else:
                print(f"📋 Other item type: {item.type}")
    print("\n------------------------------")
    print("[Final Output from RunResultStreaming]")
    print("------------------------------")
    print(run_result_stream.final_output)

async def analyze_event_types():
    """
    Analyze different types of streaming events and item types.
    """
    print("\n==============================")
    print(" 03_run_item_events.py Analysis: Event Types")
    print("==============================\n")

    result = Runner.run_streamed(joke_agent, "Give me 2 jokes")

    event_counts = {}
    item_types = {}

    async for event in result.stream_events():
        # Count event types
        event_type = event.type
        event_counts[event_type] = event_counts.get(event_type, 0) + 1

        # Count item types for run_item_stream_events
        if event_type == "run_item_stream_event":
            item_type = event.item.type
            item_types[item_type] = item_types.get(item_type, 0) + 1

    print("Event type counts:")
    for event_type, count in event_counts.items():
        print(f"  {event_type}: {count}")

    print("\nItem type counts (within run_item_stream_events):")
    for item_type, count in item_types.items():
        print(f"  {item_type}: {count}")

async def stream_with_progress_updates():
    """
    Stream agent events with user-friendly progress updates.
    """
    print("\n==============================")
    print(" 03_run_item_events.py Demo: Progress Updates")
    print("==============================\n")

    user_input = "Tell me some jokes!"
    result: RunResultStreaming = Runner.run_streamed(
        starting_agent=joke_agent,
        input=user_input
    )

    print("🚀 Starting joke generation...\n")

    async for event in result.stream_events():
        if event.type == "raw_response_event":
            continue  # Skip raw tokens

        elif event.type == "agent_updated_stream_event":
            print(f"🔄 Switched to agent: {event.new_agent.name}")

        elif event.type == "run_item_stream_event":
            item = event.item

            if item.type == "tool_call_item":
                print("🎲 Deciding how many jokes to tell...")

            elif item.type == "tool_call_output_item":
                print(f"📊 Will tell {item.output} jokes")

            elif item.type == "message_output_item":
                print("📝 Generating jokes...")

    print("\n🎉 All done!")
    print(f"🧠 Final Output:\n{result.final_output}")

# =========================
# Main Entry Point
# =========================

async def main():
    """
    Main entry point for the run item events streaming demo.
    """
    await stream_hight_level_events()
    await analyze_event_types()
    await stream_with_progress_updates()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\n❌ Exception occurred: {e}")
        print(f"Error type: {type(e).__name__}")
