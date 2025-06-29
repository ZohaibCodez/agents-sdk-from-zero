"""
Streaming Agent Runner Demonstration

This module demonstrates how to use the streaming runner functionality
to execute agent tasks with real-time event streaming. It showcases a
weather agent that provides streaming responses and uses function tools.

Author: Zohaib Khan
"""

import os
import asyncio
from typing import Optional
from dotenv import load_dotenv, find_dotenv

from agents import (
    Agent,
    Runner,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    set_tracing_disabled,
    function_tool,
    RunResultStreaming,
)
from pprint import pprint

# Load environment variables and disable tracing for cleaner output
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


@function_tool
async def get_weather(location: str) -> str:
    """
    Get weather information for a specific location.

    Args:
        location: The city or location to get weather for.

    Returns:
        str: Weather information for the specified location.
    """
    print(f"🌤️  Weather tool called for: {location}")

    # Simulate weather data retrieval
    weather_info = f"The weather in {location} is cloudy and rainy"

    print(f"📊 Weather data: {weather_info}")
    return weather_info


async def run_streaming_demo() -> None:
    """
    Demonstrate streaming agent execution with real-time event processing.

    This function creates a weather agent and runs a sample query
    to showcase the streaming execution pattern with debug output.
    """
    print("🤖 Creating streaming weather agent...")

    # Create the streaming weather agent
    streaming_agent: Agent = Agent(
        name="Weather Streaming Agent",
        instructions=(
            "You are a helpful weather assistant that can provide weather information. "
            "When asked for weather, use the get_weather tool to retrieve current conditions. "
            "Politely explain the weather after getting it and provide additional context "
            "about what the weather means for daily activities."
        ),
        model=model,
        tools=[get_weather],
    )

    print("✅ Streaming weather agent created successfully")
    print("🎬 Starting streaming agent demonstration\n")

    # Sample query for demonstration
    sample_query = "What's the weather like in Lahore today?"

    print(f"📝 Query: {sample_query}")
    print("=" * 60)
    print("📡 STREAMING EVENTS:")
    print("=" * 60)

    try:
        print(f"🚀 Starting streaming execution...\n")

        # Execute the agent with streaming
        run_result: RunResultStreaming = Runner.run_streamed(
            starting_agent=streaming_agent, input=sample_query
        )

        print("⏳ Processing streaming events...\n")

        # Process streaming events in real-time
        event_count = 0
        current_text = ""

        async for event in run_result.stream_events():
            event_count += 1
            print("\n" + "=" * 60)
            print(f"[EVENT {event_count}] : {type(event).__name__}")
            print("=" * 60 + "\n")
            pprint(event)
        print("\n" + "=" * 60)
        print("📊 FINAL RESULTS:")
        print("=" * 60)

        print(f"\n🎯 Final Response:")
        print(f"   {run_result.final_output}")

        print(f"\n📈 Summary:")
        print(f"   • Total Events: {event_count}")
        print(f"   • New Items: {len(run_result.new_items)}")

        print(f"\n📦 Generated Items:")
        for i, item in enumerate(run_result.new_items, 1):
            item_type = type(item).__name__.replace("Item", "")
            print(f"   {i}. {item_type}")

        print(f"\n✅ Streaming completed successfully!")

    except Exception as e:
        print(f"❌ Error during streaming: {e}")


if __name__ == "__main__":
    asyncio.run(run_streaming_demo())
