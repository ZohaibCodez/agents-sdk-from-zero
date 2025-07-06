"""
02_run_result_streaming_basics.py

Demonstrates basic attributes and methods of RunResultStreaming from the OpenAI Agents SDK.

Features demonstrated:
- Iterating through stream_events()
- Accessing final_output, last_agent, new_items after stream completion
- Using to_input_list() after stream completion
- Checking is_complete

Based on:
- https://openai.github.io/openai-agents-python/results/
- https://openai.github.io/openai-agents-python/ref/result/#agents.result.RunResultStreaming
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
from pprint import pprint

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
# Main Demo Function
# =========================

async def run_result_streaming_demo():
    """
    Demonstrates the basic usage of RunResultStreaming attributes and methods.
    - Streams agent output and prints all key RunResultStreaming attributes after completion.
    - Shows how to use to_input_list() for follow-up turns.
    - Prints new_items, last_agent, and input handling for clarity.
    """
    print("\n==============================")
    print(" 02_run_result_streaming_basics.py Demo")
    print("==============================\n")

    joker: Agent = Agent(
        name="Joker",
        instructions="Act as a joker.",
        model=model,
    )

    sample_query: str = "Tell me a story"
    print(f"[User Input]\n{sample_query}\n")

    try:
        run_result: RunResultStreaming = Runner.run_streamed(joker, sample_query)
        full_streamed_text: str = ""
        print("------------------------------")
        print("[Streaming Events]")
        print("------------------------------\n")
        # Stream the agent's response and collect the full text
        async for event in run_result.stream_events():
            if event.type == "raw_response_event" and isinstance(
                event.data, ResponseTextDeltaEvent
            ):
                text_chunk: str = event.data.delta
                full_streamed_text += text_chunk
                print(f"[Text Chunk]: {text_chunk}")
            else:
                print(f"[Other Event]: {str(event)[:100]}...")
        print("\n[All Collected Streamed Chunks]:")
        print(full_streamed_text)

        print("\n------------------------------")
        print("[RunResultStreaming Attributes After Completion]")
        print("------------------------------\n")

        # 1. run_result.is_complete --> True
        print(f"1️⃣  Is Complete: {run_result.is_complete}\n")

        # 2. run_result.final_output
        print("2️⃣  Final Output:")
        print("------------------")
        print(run_result.final_output)
        print(f"(Type: {type(run_result.final_output)})\n")

        # 3. run_result.last_agent
        print("3️⃣  Last Agent:")
        print("------------------")
        print(f"Name: {run_result.last_agent.name}")
        print(f"Type: {type(run_result.last_agent)}\n")

        # 4. New Items --> Items generated during agent run like new messages, tool calls and their outputs, etc.
        print("4️⃣  New Items:")
        print("------------------")
        print(f"Count: {len(run_result.new_items)}")
        for i, item in enumerate(run_result.new_items):
            print(f"   - Item {i+1}:")
            pprint(item)
            print(f"     (Type: {type(item)})\n")

        # 5. run_result.to_input_list()
        print("5️⃣  Input List for Next Turn (from to_input_list()):")
        print("------------------")
        input_list = run_result.to_input_list()
        print(f"Count: {len(input_list)}")
        for i, item_dict in enumerate(input_list):
            print(f"   - Item {i+1}:")
            pprint(item_dict)
            print(f"     (Type: {type(item_dict)})\n")

    except Exception as e:
        print(f"\n❌ An error occurred: {e}")

# =========================
# Main Entry Point
# =========================

if __name__ == "__main__":
    try:
        asyncio.run(run_result_streaming_demo())
    except Exception as e:
        print(f"\n❌ Exception occurred: {e}")
        print(f"Error type: {type(e).__name__}")
