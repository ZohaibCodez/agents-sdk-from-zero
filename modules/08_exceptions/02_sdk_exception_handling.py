"""
modules/08_exceptions/02_sdk_exception_handling.py

SDK Exception Handling Demonstration

Description:
    Demonstrates how to handle and report exceptions (especially UserError) in the OpenAI Agents SDK. Intentionally triggers a type error by passing an invalid tool to an agent, showing how to catch and report SDK exceptions in a robust, educational, and production-ready way.

Features:
    - Demonstrates robust exception handling for SDK errors
    - Visually distinct, clearly labeled output with banners and emoji
    - Educational comments and section headers
    - Explains intentional type error for demonstration

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
from agents import Agent, OpenAIChatCompletionsModel, Runner, function_tool, exceptions
from dotenv import find_dotenv, load_dotenv
from openai import AsyncOpenAI

# =========================
# 1. Environment & Model Setup
# =========================

load_dotenv(find_dotenv())

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
# 2. Exception Handling Demo
# =========================

async def trigger_user_error_with_dict_tool():
    """
    Demonstrates how the SDK raises and handles a UserError when an invalid tool is passed to an Agent.
    This intentionally triggers a type error by passing a dict instead of a Tool object to the tools list.
    The linter warning is expected and educational: do not "fix" this error, as it is the point of the demo.
    """
    print("\n==============================")
    print("🧪 [DEMO] Exception Handling: Invalid Tool Setup")
    print("==============================\n")
    try:
        # Intentionally pass an invalid tool (dict instead of Tool object)
        # This will raise a type error or UserError in the SDK
        agent = Agent(
            name="InvalidToolAgent",
            instructions="Testing invalid tool setup.",
            tools=[{"fake": "tool"}],  # <-- Intentional error for demonstration
            model=model
        )
        print("\n📝 [INPUT] Created agent with invalid tool: tools=[{'fake': 'tool'}]\n")
        runner = await Runner.run(agent, "Test input")
        print(f"🤖 [OUTPUT] Runner result: {runner.final_output}")
    except exceptions.UserError as e:
        print(f"\n❗ [EXCEPTION] Caught UserError as expected: {e}\n")
    except Exception as e:
        print(f"\n❌ [ERROR] Unexpected error: {type(e).__name__} — {e}\n")
    else:
        print("\n⚠️ [WARNING] No exception was raised, which is unexpected for this demo.\n")

# =========================
# 3. Main Entrypoint
# =========================

async def main():
    """
    Run the exception handling demonstration.
    """
    print("\n==================================================")
    print("🧪 Running SDK Exception Handling Demo")
    print("==================================================\n")
    await trigger_user_error_with_dict_tool()
    print("\n==================================================")
    print("✅ [COMPLETE] Exception handling demo finished!")
    print("==================================================\n")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\n❌ [ERROR] Unhandled exception in main: {e}\n")