"""
modules/08_exceptions/04_max_turns_exceeded.py

MaxTurnsExceeded Exception Demonstration

Description:
    Demonstrates how to handle and report MaxTurnsExceeded exceptions in the OpenAI Agents SDK. Intentionally triggers a MaxTurnsExceeded error by running an agent with max_turns=0, showing how to catch and report SDK exceptions in a robust, educational, and production-ready way.

Features:
    - Demonstrates robust exception handling for MaxTurnsExceeded
    - Visually distinct, clearly labeled output with banners and emoji
    - Educational comments and section headers
    - Explains intentional MaxTurnsExceeded error for demonstration

Environment Variables:
    - GEMINI_API_KEY: API key for Gemini model (required)

Author:
    Zohaib Khan

References:
    - https://github.com/openai/agents-sdk
    - https://platform.openai.com/docs/agents
    - https://github.com/openai/openai-python
"""
import asyncio
import os
from agents import (
    Agent,
    OpenAIChatCompletionsModel,
    Runner,
    function_tool,
    exceptions,
    set_tracing_disabled,
)
from dotenv import find_dotenv, load_dotenv
from openai import AsyncOpenAI
import logging

# =========================
# 1. Environment & Model Setup
# =========================

load_dotenv(find_dotenv())
set_tracing_disabled(True)

# Configuration constants
GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")
GEMINI_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_MODEL_NAME: str = "gemini-2.0-flash"

# Validate API key
if not GEMINI_API_KEY:
    print("\n❌ [ERROR] GEMINI_API_KEY environment variable is required but not found.\n")
    raise ValueError("GEMINI_API_KEY environment variable is required but not found.")

print("\n==============================")
print(f"🚀 Initializing Gemini client with model: {GEMINI_MODEL_NAME}")
print("==============================\n")

# Initialize external OpenAI client for Gemini
external_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url=GEMINI_BASE_URL,
)

# Configure the chat completions model
model = OpenAIChatCompletionsModel(
    openai_client=external_client, model=GEMINI_MODEL_NAME
)

print(f"✅ [INFO] Model configured successfully\n")

# =========================
# 2. MaxTurnsExceeded Exception Demo
# =========================

async def demo_max_turns_exceeded():
    """
    Demonstrates how the SDK raises and handles a MaxTurnsExceeded exception when max_turns=0 is set.
    This intentionally triggers the error for educational purposes.
    """
    print("\n==============================")
    print("🧪 [DEMO] Exception Handling: MaxTurnsExceeded (max_turns=0)")
    print("==============================\n")
    try:
        # Intentionally run the agent with max_turns=0 to trigger MaxTurnsExceeded
        print("📝 [INPUT] Running agent with max_turns=0 (intentional error)\n")
        agent = Agent(
            name="MisconfiguredAgent",
            instructions="You are a helpful assitant.",
        )
        response = await Runner.run(agent, "Hi", max_turns=0)
        print(f"🤖 [OUTPUT] Runner result: {response.final_output}")
    except exceptions.MaxTurnsExceeded as e:
        print(f"\n❗ [EXCEPTION] Caught MaxTurnsExceeded as expected: {e}\n")
    except Exception as e:
        print(f"\n❌ [ERROR] Unexpected error in demo_max_turns_exceeded: {type(e).__name__} - {e}\n")
    else:
        print("\n⚠️ [WARNING] No exception was raised, which is unexpected for this demo.\n")

# =========================
# 3. Main Entrypoint
# =========================

async def main():
    """
    Run the MaxTurnsExceeded exception handling demonstration.
    """
    print("\n==================================================")
    print("🧪 Running MaxTurnsExceeded Exception Handling Demo")
    print("==================================================\n")
    await demo_max_turns_exceeded()
    print("\n==================================================")
    print("✅ [COMPLETE] MaxTurnsExceeded exception handling demo finished!")
    print("==================================================\n")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\n❌ [ERROR] Unhandled exception in main: {e}\n")
