"""
04_streaming_with_handoffs.py

Demonstrates streaming with agent handoffs and transitions in real-time.

This module showcases how to:
- Set up multiple specialized agents with different capabilities
- Configure agent handoffs for seamless transitions
- Stream and monitor agent handoff events in real-time
- Handle different types of streaming events (tool calls, handoffs, messages)
- Process agent transitions and tool outputs during streaming

Core Concept: Stream agent handoffs and see how agent transitions work in real-time,
providing visibility into the handoff process and agent interactions.

Based on:
- https://openai.github.io/openai-agents-python/streaming/
- https://openai.github.io/openai-agents-python/handoffs/

Environment variables:
    - GEMINI_API_KEY: Required for Gemini model access.

Author: Zohaib Khan
"""

import asyncio
import os
from agents.handoffs import Handoff, handoff
from dotenv import load_dotenv, find_dotenv
from openai import AsyncOpenAI
from openai.types.responses import ResponseTextDeltaEvent
from agents import (
    Agent,
    Runner,
    OpenAIChatCompletionsModel,
    RunResultStreaming,
    set_tracing_disabled,
    function_tool,
    ItemHelpers,
)
import random

# =========================
# Environment & Model Setup
# =========================

load_dotenv(find_dotenv())
set_tracing_disabled(True)

# Configuration constants
GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")
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
# Tool Definitions
# =========================


@function_tool
def get_random_number() -> int:
    """
    Generate a random number between 0 and 10.

    Returns:
        int: A random integer between 0 and 10 (inclusive)
    """
    return random.randint(0, 10)


@function_tool
def get_current_weather(location: str) -> str:
    """
    Get the current weather for a specified location.

    Args:
        location (str): The location to get weather information for

    Returns:
        str: Weather information for the specified location
    """
    if location.lower() == "karachi":
        return f"Weather in {location} is Cloudy"
    elif location.lower() == "islamabad":
        return f"Weather in {location} is Sunny"
    elif location.lower() == "lahore":
        return f"Weather in {location} is Rainy"
    else:
        return f"Weather for {location} is unknown"


# =========================
# Agent Definitions
# =========================

joke_agent: Agent = Agent(
    name="Joker Agent",
    instructions="Act as a joker. Use get_random_number to get a number. And for that number tell that many jokes.",
    tools=[get_random_number],
    model=model,
)

weather_agent: Agent = Agent(
    name="Weather Agent",
    instructions="You are a weather agent. Use weather tools for weather related queries and handle every weather related query.",
    tools=[get_current_weather],
    model=model,
)

coordinator: Agent = Agent(
    name="Coordinator Agent",
    instructions="""
    You are a helpful coordinator. 
    - For joke questions, hand off to JokeSpecialist
    - For weather queries, hand off to WeatherSpecialist
    - For other questions, handle them yourself""",
    model=model,
    handoffs=[
        handoff(
            weather_agent,
            tool_description_override="Handoff Weather queries to the weather_agent",
        ),
        handoff(
            joke_agent,
            tool_description_override="Handoff joker related queries to joke_agent",
        ),
    ],
)

# =========================
# Streaming Functions
# =========================


async def stream_joke_handoff():
    """
    Demonstrate streaming with agent handoffs for joke-related queries.

    This function:
    - Initiates a streaming conversation with the joke agent
    - Monitors real-time events during the conversation
    - Tracks agent transitions and tool usage
    - Displays formatted output for different event types
    """
    sample_query: str = "Hi Joker"

    print("=" * 60)
    print("🎭 STREAMING JOKER HANDOFF DEMO")
    print("=" * 60)
    print(f"📝 User Input: {sample_query}")
    print("-" * 60)

    try:
        result = Runner.run_streamed(coordinator, sample_query)
        current_agent: str = "Unknown"

        async for event in result.stream_events():
            if event.type == "raw_response_event":
                continue
            elif event.type == "agent_updated_stream_event":
                current_agent = event.new_agent.name
                print(f"🔄 Agent switched to: {current_agent}")
            elif event.type == "run_item_stream_event":
                item = event.item
                if item.type == "handoff_call_item":
                    print(f"🤝 Handing off ... ")
                elif item.type == "handoff_output_item":
                    print(
                        f"📤 Switching from {item.source_agent.name} to {item.target_agent.name}"
                    )
                elif item.type == "tool_call_item":
                    # Fix for linter error - use getattr to safely access name attribute
                    tool_name = getattr(item.raw_item, "name", "Unknown Tool")
                    print(f"🔧 Tool called: {tool_name}")
                elif item.type == "tool_call_output_item":
                    print(f"📊 Tool Output: {item.output}")
                elif item.type == "message_output_item":
                    message_text = ItemHelpers.text_message_output(item)
                    print(f"💬 {current_agent} says: {message_text}")

        print("-" * 60)
        print("✅ Joker handoff demo completed successfully")

    except Exception as e:
        print(f"❌ Error during joker handoff: {e}")
        raise


async def stream_weather_handoff():
    """
    Demonstrate streaming with agent handoffs for weather-related queries.

    This function:
    - Initiates a streaming conversation with the weather agent
    - Monitors real-time events during the conversation
    - Tracks agent transitions and tool usage
    - Displays formatted output for different event types
    """
    sample_query: str = "Tell me a weather about Lahore"

    print("=" * 60)
    print("🌤️ STREAMING WEATHER HANDOFF DEMO")
    print("=" * 60)
    print(f"📝 User Input: {sample_query}")
    print("-" * 60)

    try:
        result = Runner.run_streamed(coordinator, sample_query)
        current_agent: str = "Unknown"

        async for event in result.stream_events():
            if event.type == "raw_response_event":
                continue
            elif event.type == "agent_updated_stream_event":
                current_agent = event.new_agent.name
                print(f"🔄 Agent switched to: {current_agent}")
            elif event.type == "run_item_stream_event":
                item = event.item
                if item.type == "handoff_call_item":
                    print(f"🤝 Handing off ... ")
                elif item.type == "handoff_output_item":
                    print(
                        f"📤 Switching from {item.source_agent.name} to {item.target_agent.name}"
                    )
                elif item.type == "tool_call_item":
                    # Fix for linter error - use getattr to safely access name attribute
                    tool_name = getattr(item.raw_item, "name", "Unknown Tool")
                    print(f"🔧 Tool called: {tool_name}")
                elif item.type == "tool_call_output_item":
                    print(f"📊 Tool Output: {item.output}")
                elif item.type == "message_output_item":
                    message_text = ItemHelpers.text_message_output(item)
                    print(f"💬 {current_agent} says: {message_text}")

        print("-" * 60)
        print("✅ Weather handoff demo completed successfully")

    except Exception as e:
        print(f"❌ Error during weather handoff: {e}")
        raise


async def stream_no_handoff_demo():
    """
    Demonstrate streaming with no agent handoffs for general queries.

    This function:
    - Initiates a streaming conversation with the coordinator agent
    - Shows how the coordinator handles general queries without handoffs
    - Monitors real-time events during the conversation
    - Displays formatted output for different event types
    """
    sample_query: str = "Hi"

    print("=" * 60)
    print("👋 STREAMING NO HANDOFF DEMO")
    print("=" * 60)
    print(f"📝 User Input: {sample_query}")
    print("-" * 60)

    try:
        result = Runner.run_streamed(coordinator, sample_query)
        current_agent: str = "Unknown"

        async for event in result.stream_events():
            if event.type == "raw_response_event":
                continue
            elif event.type == "agent_updated_stream_event":
                current_agent = event.new_agent.name
                print(f"🔄 Agent switched to: {current_agent}")
            elif event.type == "run_item_stream_event":
                item = event.item
                if item.type == "handoff_call_item":
                    print(f"🤝 Handing off ... ")
                elif item.type == "handoff_output_item":
                    print(
                        f"📤 Switching from {item.source_agent.name} to {item.target_agent.name}"
                    )
                elif item.type == "tool_call_item":
                    # Fix for linter error - use getattr to safely access name attribute
                    tool_name = getattr(item.raw_item, "name", "Unknown Tool")
                    print(f"🔧 Tool called: {tool_name}")
                elif item.type == "tool_call_output_item":
                    print(f"📊 Tool Output: {item.output}")
                elif item.type == "message_output_item":
                    message_text = ItemHelpers.text_message_output(item)
                    print(f"💬 {current_agent} says: {message_text}")

        print("-" * 60)
        print("✅ No handoff demo completed successfully")

    except Exception as e:
        print(f"❌ Error during no handoff demo: {e}")
        raise


async def analyze_stream_handoff_patterns():
    """
    Analyze streaming handoff patterns across different test cases.
    
    This function:
    - Tests multiple scenarios to understand handoff behavior
    - Tracks handoff counts and agent usage patterns
    - Provides comprehensive analysis of streaming events
    - Demonstrates how different queries trigger different handoff patterns
    """
    test_cases: list[tuple[str, str]] = [
        ("Joke", "Tell me some programming jokes"),
        ("Weather", "What's weather in Lahore"),
        ("General", "What is the capital of Germany?"),
    ]
    
    print("=" * 60)
    print("📊 STREAMING HANDOFF PATTERN ANALYSIS")
    print("=" * 60)
    
    for case_type, user_input in test_cases:
        print(f"\n{'='*40}")
        print(f"🧪 {case_type.upper()} CASE ANALYSIS")
        print(f"📝 User Input: '{user_input}'")
        print(f"{'='*40}")

        try:
            result: RunResultStreaming = Runner.run_streamed(coordinator, user_input)

            handoff_count = 0
            agents_used = []
            current_agent = "Unknown"

            async for event in result.stream_events():
                if event.type == "raw_response_event":
                    continue
                elif event.type == "agent_updated_stream_event":
                    current_agent = event.new_agent.name
                    if current_agent not in agents_used:
                        agents_used.append(current_agent)
                    print(f"🔄 Agent switched to: {current_agent}")
                elif event.type == "run_item_stream_event":
                    item = event.item
                    if item.type == "handoff_call_item":
                        handoff_count += 1
                        print(f"🤝 Handing off ... ")
                    elif item.type == "handoff_output_item":
                        print(
                            f"📤 Switching from {item.source_agent.name} to {item.target_agent.name}"
                        )
                    elif item.type == "tool_call_item":
                        # Fix for linter error - use getattr to safely access name attribute
                        tool_name = getattr(item.raw_item, "name", "Unknown Tool")
                        print(f"🔧 Tool called: {tool_name}")
                    elif item.type == "tool_call_output_item":
                        print(f"📊 Tool Output: {item.output}")
                    elif item.type == "message_output_item":
                        message_text = ItemHelpers.text_message_output(item)
                        print(f"💬 {current_agent} says: {message_text}")

            # Print analysis summary for this case
            print(f"\n📈 {case_type} Case Summary:")
            print(f"   • Total Handoffs: {handoff_count}")
            print(f"   • Agents Used: {', '.join(agents_used)}")
            print(f"   • Final Agent: {current_agent}")
            
        except Exception as e:
            print(f"❌ Error during {case_type} case analysis: {e}")
            continue

    print(f"\n{'='*60}")
    print("✅ Handoff pattern analysis completed successfully")
    print(f"{'='*60}")


# =========================
# Main Execution
# =========================


async def main():
    """
    Main execution function that runs the streaming handoff demonstrations.

    This function orchestrates the demos and handles any top-level exceptions.
    You can uncomment different demo functions to test various scenarios.
    """
    try:
        # Choose which demo to run
        # await stream_joke_handoff()      # Uncomment to run joke handoff demo
        # await stream_weather_handoff()   # Uncomment to run weather handoff demo
        # await stream_no_handoff_demo()   # Uncomment to run no handoff demo
        await analyze_stream_handoff_patterns()  # Currently running pattern analysis

    except Exception as e:
        print(f"❌ Fatal error in main execution: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
