"""
06_runner_rcontext.py

Demonstrates advanced usage of the OpenAI Agents SDK Runner with RunContextWrapper,
including tool calls, max turn handling, and input type flexibility.

Features demonstrated:
- Environment and model setup with debug output
- Tool function registration and debug
- Agent creation and debug
- Handling max_turns and MaxTurnsExceeded
- Supporting multiple input types (string, chat, history)
- Debug logging for troubleshooting

Environment variables:
    - GEMINI_API_KEY: Required for Gemini model access.

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
    RunResult,
    ToolCallOutputItem,
    RunContextWrapper,
    Handoff,
    MaxTurnsExceeded,
)

# =========================
# Environment & Model Setup
# =========================

# Load environment variables and disable tracing for cleaner output
load_dotenv(find_dotenv())
set_tracing_disabled(True)

print("🔧 DEBUG: Environment setup started")

# Configuration constants
GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
GEMINI_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_MODEL_NAME: str = "gemini-2.0-flash"

print(f"🔧 DEBUG: Configuration loaded - Model: {GEMINI_MODEL_NAME}")

# Validate API key
if not GEMINI_API_KEY:
    print("❌ GEMINI_API_KEY environment variable is required but not found.")
    raise ValueError("GEMINI_API_KEY environment variable is required but not found.")

print(f"✅ DEBUG: API key validation successful")
print(f"🚀 Initializing Gemini client with model: {GEMINI_MODEL_NAME}")

# Initialize external OpenAI client for Gemini
external_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url=GEMINI_BASE_URL,
)

print(f"✅ DEBUG: External client initialized successfully")

# Configure the chat completions model
model = OpenAIChatCompletionsModel(
    openai_client=external_client, model=GEMINI_MODEL_NAME
)

print(f"✅ DEBUG: Model configuration completed\n")


@function_tool
def get_current_weather(city: str) -> str:
    """Get the current weather for a given city."""
    print(f"🔧 DEBUG: Tool 'get_current_weather' called for city: {city}")
    if city == "New York":
        return "The weather in New York is sunny."
    elif city == "London":
        return "The weather in London is cloudy."
    else:
        return f"The weather in {city} is unknown."


@function_tool
def get_city_fact(city: str) -> str:
    """Get a fact about a given city."""
    print(f"🔧 DEBUG: Tool 'get_city_fact' called for city: {city}")
    if city.lower() == "new york":
        return "New York is the city that never sleeps."
    elif city.lower() == "london":
        return "London is the city that never sleeps."
    else:
        return f"The fact about {city} is unknown."


async def demo_loop_final_output():
    """Demo: Simple agent greeting."""
    print("\n🔄 Starting demo_loop_final_output...")
    greet_agent: Agent = Agent(
        name="Greet Agent", instructions="You have to greet the user.", model=model
    )
    print(f"🤖 Created agent: {greet_agent.name}")
    result: RunResult = await Runner.run(greet_agent, "Hello, how are you?")
    print(f"🤖 Agent Response: {result.final_output}")


async def demo_loop_with_toolcall():
    """Demo: Agent with tool call for weather."""
    print("\n🔄 Starting demo_loop_with_toolcall...")
    weather_agent: Agent = Agent(
        name="Weather Agent",
        instructions="You have to get the current weather for a given city.",
        model=model,
        tools=[get_current_weather],
    )
    print(f"🤖 Created agent: {weather_agent.name}")
    result: RunResult = await Runner.run(
        weather_agent, "What is the weather in New York?"
    )
    print(f"🤖 Agent Response: {result.final_output}")


async def demo_max_turns():
    """Demo: max_turns and MaxTurnsExceeded exception."""
    print("\n🚨 Demo: max_turns and MaxTurnsExceeded 🚨")
    looping_agent: Agent = Agent(
        name="Looping Agent",
        instructions="Always use the get_city_fact tool for 'New York'. "
        "Do not provide a final answer, just keep researching.",
        tools=[get_city_fact],
        model=model,
    )
    print(f"🤖 Created agent: {looping_agent.name}")
    try:
        result = await Runner.run(
            looping_agent,
            input="Fact about New York",
            max_turns=1,  # Limit to prevent infinite loop
        )
        print("🔚 Final Output:", result.final_output)
    except MaxTurnsExceeded as e:
        print("❌ Caught MaxTurnsExceeded:", e)
    except Exception as e:
        print("❌ Unexpected Error:", e)


async def demo_input_types():
    """Demo: Agent input types (string, chat, history)."""
    echo_agent = Agent(
        name="EchoAgent",
        instructions="Repeat the user's message exactly.",
        model=model,
    )
    print("\n🧪 Demo: Input Types")
    # --- 1. String input ---
    print("\n1️⃣ String Input:")
    simple_input = "Echo this back to me!"
    result1 = await Runner.run(echo_agent, simple_input)
    print(f"📨 Input: {simple_input}")
    print(f"🤖 Output: {result1.final_output}")

    # --- 2. List of OpenAI-style chat messages ---
    print("\n2️⃣ List of Input Items (Chat Format):")
    chat_input = [
        {"role": "system", "content": "You are an assistant that echoes user input."},
        {"role": "user", "content": "Hello from the chat message!"},
    ]
    result2 = await Runner.run(echo_agent, chat_input)
    print("📨 Input:")
    pprint(chat_input)
    print(f"🤖 Output: {result2.final_output}")

    # --- 3. Simulated history with follow-up input ---
    print("\n3️⃣ Simulated Conversation History:")
    history_input = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": "Thanks! And Germany?"},
    ]
    result3 = await Runner.run(echo_agent, history_input)
    print("📨 Input:")
    pprint(history_input)
    print(f"🤖 Output: {result3.final_output}")


async def main():
    """Main entry point for all demos."""
    print("\n🚀 Starting 06_runner_rcontext.py demo suite")
    await demo_loop_final_output()
    await demo_loop_with_toolcall()
    await demo_max_turns()
    await demo_input_types()


if __name__ == "__main__":
    asyncio.run(main())
