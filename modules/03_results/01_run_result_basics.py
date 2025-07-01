"""
01_run_result_basics.py

Demonstrates basic attributes and methods of RunResult from the OpenAI Agents SDK.

Features demonstrated:
- Accessing final_output
- Accessing last_agent
- Using to_input_list() for subsequent turns
- Inspecting new_items (content and type)
- Accessing the original input

Based on:
- https://openai.github.io/openai-agents-python/results/
- https://openai.github.io/openai-agents-python/ref/result/#agents.result.RunResultBase

Environment variables:
    - GEMINI_API_KEY: Required for Gemini model access.

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
    RunResult,
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
# Agent Setup
# =========================

basic_agent: Agent = Agent(
    name="Greet Agent",
    instructions="You are a simple Greet agent that greets the user and handle quereies with emojies.",
    model=model,
)

# =========================
# Main Demo Function
# =========================

async def main():
    """
    Demonstrates the basic usage of RunResult attributes and methods.
    - Runs an initial turn and prints all key RunResult attributes.
    - Shows how to use to_input_list() for follow-up turns.
    - Prints new_items, last_agent, and input handling for clarity.
    """
    print("\n==============================")
    print("   01_run_result_basics.py Demo")
    print("==============================\n")

    sample_query_1: str = (
        "Hello! My name is zohaib. I am a Data science student currently undergraduate and also doing some learning stuff in currently emerging tech named Agentic AI."
    )
    print(f"[User Input - Turn 1]\n{sample_query_1}\n")

    try:
        run_result_1: RunResult = await Runner.run(
            starting_agent=basic_agent, input=sample_query_1
        )

        print("------------------------------")
        print("[RunResult Attributes - Turn 1]")
        print("------------------------------\n")

        # 1. Final Output:
        final_output = run_result_1.final_output
        print("1️⃣  Final Output:")
        print("------------------")
        print(final_output)
        print(f"(Type: {type(final_output)})\n")

        # 2. Last Agent
        last_agent = run_result_1.last_agent
        print("2️⃣  Last Agent:")
        print("------------------")
        print(f"Name: {last_agent.name}")
        print(f"Type: {type(last_agent)}\n")

        # 3. New Items (messages, tool calls, etc.)
        new_items = run_result_1.new_items
        print("3️⃣  New Items:")
        print("------------------")
        print(f"Count: {len(new_items)}")
        if new_items:
            for i, item in enumerate(new_items):
                print(f"   - Item {i+1}:")
                pprint(item)
                print(f"     (Type: {type(item)})\n")
        else:
            print("   No new items generated in this simple run (besides final_output which is part of it).\n")

        # 4. Original Input
        original_input: str = run_result_1.input
        print("4️⃣  Original Input:")
        print("------------------")
        print(original_input)
        print(f"(Type: {type(original_input)})\n")

        # 5. Inputs for the next turn (Using .to_input_list())
        input_list = run_result_1.to_input_list()
        print("5️⃣  Input List for Next Turn (from to_input_list()):")
        print("------------------")
        print(f"Count: {len(input_list)}")
        for i, item_dict in enumerate(input_list):
            print(f"   - Item {i+1}:")
            pprint(item_dict)
            print(f"     (Type: {type(item_dict)})\n")

        # Adding a new user message for a follow-up turn
        sample_query_2 = "What is my name? Do you remember me ?"
        convo = input_list + [{"role": "user", "content": sample_query_2}]
        print("6️⃣  New User Message for Turn 2:")
        print("------------------")
        print(f"User: {sample_query_2}\n")
        print("Combined History for Turn 2:")
        print(f"Count: {len(convo)}")
        for i, item_dict in enumerate(convo):
            print(f"   - Item {i+1}:")
            pprint(item_dict)
            print()

    except Exception as e:
        print(f"\n❌ An error occurred: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\n❌ Exception occurred: {e}")
        print(f"Error type: {type(e).__name__}")
