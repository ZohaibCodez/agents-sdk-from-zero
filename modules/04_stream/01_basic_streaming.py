"""
01_basic_streaming.py

Simple introduction to streaming responses with OpenAI Agents SDK.

Features demonstrated:
- Streaming responses in real-time as they're generated
- Printing streamed text as it arrives
- Comparing streamed text with final output

Based on:
- https://openai.github.io/openai-agents-python/streaming/

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
# Streaming Demo Function
# =========================


async def demo_basic_streaming():
    """
    Streams agent output in real-time and prints the streamed text as it arrives.
    Shows the final output for comparison.
    """
    sample_query: str = "Hi joker. Tell me some amazing story."
    print("\n==============================")
    print(" 01_basic_streaming.py Demo")
    print("==============================\n")
    print(f"[User Input]\n{sample_query}\n")

    run_result_stream: RunResultStreaming = Runner.run_streamed(
        joke_agent, sample_query
    )

    print("------------------------------")
    print("[Streaming Events]")
    print("------------------------------\n")
    async for event in run_result_stream.stream_events():
        event_type = type(event).__name__
        print(f"📡 Received event type: {event_type}")

    print("[Final Output from RunResultStreaming]")
    print("------------------------------")
    print(run_result_stream.final_output)


# =========================
# Main Entry Point
# =========================


async def main():
    """
    Main entry point for the basic streaming demo.
    """
    await demo_basic_streaming()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\n❌ Exception occurred: {e}")
        print(f"Error type: {type(e).__name__}")
