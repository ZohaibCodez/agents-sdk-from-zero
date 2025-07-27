"""
modules/08_exceptions/03_user_error_demo.py

UserError Exception Demonstration

Description:
    Demonstrates how to handle and report UserError exceptions in the OpenAI Agents SDK. Intentionally triggers a UserError by passing an invalid model parameter to an agent, showing how to catch and report SDK exceptions in a robust, educational, and production-ready way.

Features:
    - Demonstrates robust exception handling for UserError
    - Visually distinct, clearly labeled output with banners and emoji
    - Educational comments and section headers
    - Explains intentional UserError for demonstration

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
from agents import Agent, OpenAIChatCompletionsModel, Runner, function_tool, exceptions,set_tracing_disabled
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
# 2. UserError Exception Demo
# =========================

async def demo_user_error():
    """
    Demonstrates how the SDK raises and handles a UserError when an invalid model parameter is passed to an Agent.
    This intentionally triggers a UserError by passing an integer instead of a valid model or model name.
    The linter warning is expected and educational: do not "fix" this error, as it is the point of the demo.
    """
    print("\n==============================")
    print("🧪 [DEMO] Exception Handling: UserError (Invalid Model Parameter)")
    print("==============================\n")
    try:
        # Intentionally pass an invalid model (integer instead of string or ModelProvider)
        # This will raise a UserError in the SDK
        print("📝 [INPUT] Creating agent with invalid model parameter: model=12345\n")
        agent = Agent(
            name="MisconfiguredAgent",
            instructions="This agent is intentionally misconfigured.",
            tools=[], # No tools needed for this demo
            model=12345 # <-- Intentional error for demonstration
        )
        print("Agent created with invalid model parameter (UNEXPECTED - UserError should have been raised).\n")
        response = await Runner.run(agent, "Hi")
        print(f"🤖 [OUTPUT] Runner result: {response.final_output}")
    except exceptions.UserError as e:
        print(f"\n❗ [EXCEPTION] Caught UserError as expected: {e}\n")
    except Exception as e:
        print(f"\n❌ [ERROR] Unexpected error in demo_user_error: {type(e).__name__} - {e}\n")
    else:
        print("\n⚠️ [WARNING] No exception was raised, which is unexpected for this demo.\n")

# =========================
# 3. Main Entrypoint
# =========================

async def main():
    """
    Run the UserError exception handling demonstration.
    """
    print("\n==================================================")
    print("🧪 Running UserError Exception Handling Demo")
    print("==================================================\n")
    await demo_user_error()
    print("\n==================================================")
    print("✅ [COMPLETE] UserError exception handling demo finished!")
    print("==================================================\n")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\n❌ [ERROR] Unhandled exception in main: {e}\n") 