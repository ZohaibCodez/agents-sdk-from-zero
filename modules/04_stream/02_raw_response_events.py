"""
02_raw_response_events.py

Demonstrates raw response events for token-by-token text streaming using the OpenAI Agents SDK.

Features demonstrated:
- Capture text as it's generated token-by-token using raw response events
- Analyze event types and text deltas
- Print streamed text and compare with final output

Based on:
- https://openai.github.io/openai-agents-python/streaming/
- Raw response events section

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
)

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
# Agent Setup
# =========================

joke_agent: Agent = Agent(
    name="Joke Agent",
    instructions="Act as a joker. And use some emojies for better interaction.",
    model=model,
)

# =========================
# Streaming Demo Functions
# =========================

async def token_by_token_streaming():
    """
    Streams agent output token-by-token and prints the streamed text as it arrives.
    Shows the final output for comparison.
    """
    sample_query: str = "Hi joker. Tell me some amazing story."
    print("\n==============================")
    print(" 02_raw_response_events.py Demo")
    print("==============================\n")
    print(f"[User Input]\n{sample_query}\n")

    collected_text: str = ""
    run_result_stream: RunResultStreaming = Runner.run_streamed(
        joke_agent, sample_query
    )

    print("------------------------------")
    print("[Token-by-Token Streaming]")
    print("------------------------------\n")
    async for event in run_result_stream.stream_events():
        if event.type == "raw_response_event" and isinstance(
            event.data, ResponseTextDeltaEvent
        ):
            print(event.data.delta, end="", flush=True)
            collected_text += event.data.delta
    print("\n\n------------------------------")
    print("[Streamed Text Collected]")
    print("------------------------------")
    print(collected_text)
    print("\n------------------------------")
    print("[Final Output from RunResultStreaming]")
    print("------------------------------")
    print(run_result_stream.final_output)
    if collected_text.strip() == run_result_stream.final_output.strip():
        print("\n✅ Streamed text matches final output.")
    else:
        print("\n⚠️  Streamed text does NOT match final output!")

async def analyze_raw_response_events():
    """
    Analyzes raw response events, counts event types, and prints text delta statistics.
    """
    print("\n==============================")
    print(" 02_raw_response_events.py Analysis")
    print("==============================\n")
    result = Runner.run_streamed(joke_agent, "Tell me a joke about programming")

    event_counts = {}
    text_deltas = []

    async for event in result.stream_events():
        if event.type == "raw_response_event":
            event_data_type = type(event.data).__name__
            event_counts[event_data_type] = event_counts.get(event_data_type, 0) + 1

            # Collect text deltas specifically
            if isinstance(event.data, ResponseTextDeltaEvent):
                text_deltas.append(event.data.delta)

    print(f"Raw event types encountered:")
    for event_type, count in event_counts.items():
        print(f"  {event_type}: {count} events")

    print(f"\nText deltas collected: {len(text_deltas)}")
    print(f"Total text length: {sum(len(delta) for delta in text_deltas)} characters")

    if text_deltas:
        print(f"First 3 deltas: {text_deltas[:3]}")
        print(f"Last 3 deltas: {text_deltas[-3:]}")

# =========================
# Main Entry Point
# =========================

async def main():
    """
    Main entry point for the raw response events streaming demo.
    """
    # await token_by_token_streaming()
    await analyze_raw_response_events()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\n❌ Exception occurred: {e}")
        print(f"Error type: {type(e).__name__}")
