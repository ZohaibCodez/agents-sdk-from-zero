"""
04_handoff_input_filter_basics.py

Demonstrates the use of input filters in agent handoffs to control what context is passed to a support agent. Shows the difference between a standard handoff (with tool call logs) and a filtered handoff (tool calls removed).

Features:
- Professional, clear output with banners and emoji
- Input/output separation for demo cases
- Robust error handling
- Example of using `handoff_filters.remove_all_tools`
- Educational comments and output

Environment Variables:
- MISTRAL_API_KEY: API key for Mistral/OpenRouter

Author: OpenAI Agents SDK Example Maintainer
References: See SDK documentation for handoff filters and agent chaining.

Note: The last four lines of output in the terminal are due to litellm and can be ignored. This may be fixed in a future update.
"""

import asyncio
import os
from dotenv import load_dotenv, find_dotenv
from agents.extensions.models.litellm_model import LitellmModel
from agents.extensions import handoff_filters
from agents import (
    Agent,
    function_tool,
    handoff,
    Runner,
    set_tracing_disabled,
    enable_verbose_stdout_logging,  # <-- Add this import
)

# =========================
# Verbose Logging (LLM Debug Mode)
# =========================
# To see detailed message history, tool calls, and handoff context in the terminal output,
# uncomment the following line. This is very useful for debugging and educational purposes!
# enable_verbose_stdout_logging()

# =========================
# Environment & Model Setup
# =========================
load_dotenv(find_dotenv())
set_tracing_disabled(True)

MISTRAL_API_KEY: str | None = os.getenv("MISTRAL_API_KEY")
MODEL_NAME: str = "mistral/mistral-small-2506"

if not MISTRAL_API_KEY:
    print("\n❌ [ERROR] MISTRAL_API_KEY environment variable is required but not found.\n")
    raise ValueError("MISTRAL_API_KEY environment variable is required but not found.")

print("\n" + "="*60)
print(f"🚀 Initializing OpenRouter client with model: {MODEL_NAME}")
model = LitellmModel(model=MODEL_NAME, api_key=MISTRAL_API_KEY)
print(f"✅ Model configured successfully")
print("="*60 + "\n")

# ====================
# A dummy tool
# ====================
@function_tool
def say_hello(name: str) -> str:
    """
    Returns a greeting for the given name.
    Args:
        name (str): The name to greet.
    Returns:
        str: Greeting message.
    """
    return f"Hello, {name}!"

# =========================
# Test WITHOUT input filter
# =========================
async def test_without_filter():
    """
    Runs a handoff demo WITHOUT input filter (tool logs will be visible).
    """
    print("\n" + "="*60)
    print("🔎 TEST 1: WITHOUT FILTER (tool logs will be visible)")
    print("="*60)
    support_agent = Agent(
        name="Support Agent",
        instructions="You are the support agent. Answer user queries politely with emojis.",
        model=model,
    )
    main_agent = Agent(
        name="Main Agent",
        instructions="You are the main agent. Use say_hello tool, then hand off to support agent.",
        tools=[say_hello],
        model=model,
        handoffs=[
            handoff(
                agent=support_agent,
                tool_name_override="handoff_without_filter",
            ),
        ],
    )
    try:
        user_input = "Use the say_hello tool for name Alice, then transfer to support and I want support in mathematics."
        print(f"\n📝 [Input] {user_input}")
        result = await Runner.run(
            main_agent,
            input=user_input,
        )
        print(f"\n👤 [Last Agent] {getattr(getattr(result, 'last_agent', None), 'name', 'N/A')}")
        print(f"💬 [Final Output] {getattr(result, 'final_output', None)}")
    except Exception as e:
        print(f"\n❌ [ERROR in test_without_filter] {e}")

# =====================
# Test WITH input filter
# =====================
async def test_with_filter():
    """
    Runs a handoff demo WITH input filter (tool logs will be hidden).
    """
    print("\n" + "="*60)
    print("🔒 TEST 2: WITH FILTER (tool logs will be hidden)")
    print("="*60)
    support_agent = Agent(
        name="Support Agent",
        instructions="You are the support agent. Always greet the user and answer queries politely with emojis.",
        model=model,
    )
    main_agent = Agent(
        name="Main Agent",
        instructions="You are the main agent. Use say_hello tool, then hand off to support agent.",
        tools=[say_hello],
        model=model,
        handoffs=[
            handoff(
                agent=support_agent,
                tool_name_override="handoff_with_filter",
                input_filter=handoff_filters.remove_all_tools,
            ),
        ],
    )
    try:
        user_input = "Use the say_hello tool for name Alice, then transfer to support and I want support in mathematics."
        print(f"\n📝 [Input] {user_input}")
        result = await Runner.run(
            main_agent,
            input=user_input,
        )
        print(f"\n👤 [Last Agent] {getattr(getattr(result, 'last_agent', None), 'name', 'N/A')}")
        print(f"💬 [Final Output] {getattr(result, 'final_output', None)}")
    except Exception as e:
        print(f"\n❌ [ERROR in test_with_filter] {e}")

# =============
# Main Entrypoint
# =============
async def main():
    """
    Runs both demo cases: without and with input filter.
    """
    await test_without_filter()
    await test_with_filter()

if __name__ == "__main__":
    try:
        asyncio.run(main())
        print("\n✅ All tests completed successfully!\n")
    except Exception as e:
        print(f"\n❌ [FATAL ERROR] {e}\n")
