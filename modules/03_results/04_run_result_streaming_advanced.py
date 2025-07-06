"""
04_run_result_streaming_advanced.py

Demonstrates advanced RunResultStreaming usage with tools, handoffs, and complex scenarios using the OpenAI Agents SDK.

Features demonstrated:
- Streaming with tool usage and complex event types
- Real-time event processing and filtering
- Performance monitoring during streaming
- Error handling in streaming scenarios
- Memory-efficient streaming patterns

Based on:
- https://openai.github.io/openai-agents-python/results/
- https://openai.github.io/openai-agents-python/streaming/
- https://openai.github.io/openai-agents-python/tools/

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
    handoff,
    set_tracing_disabled,
    StreamEvent,
    function_tool,
)
from pprint import pprint
import time
import random

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
# Tool Functions
# =========================

@function_tool
def calculate(a: int, b: int) -> int:
    """Tool to add two numbers."""
    return a + b

@function_tool
def get_random_number() -> int:
    """Tool to get a random integer between 1 and 10."""
    return random.randint(1, 10)

# =========================
# Specialized Agents
# =========================

math_agent: Agent = Agent(
    name="Math Agent",
    instructions="Act as a mathematician. Solve numeric problems using the calculate tool.",
    model=model,
    tools=[calculate],
)

joke_agent: Agent = Agent(
    name="Joke Agent",
    instructions="Act as a joker. Use get_random_number to get a number. And for that number tell that many jokes.",
    model=model,
    tools=[get_random_number],
)

coordinator_agent: Agent = Agent(
    name="Coordinator Agent",
    instructions="""Act as a coordinator agent.
    - For Math Queries, handoff to math_agent
    - For Joke Queries, handoff to joke_agent
    - For other queries, handle them yourself
""",
    model=model,
    handoffs=[
        handoff(math_agent, tool_description_override="Solve math problems"),
        handoff(
            joke_agent, tool_description_override="Tell jokes using a random number"
        ),
    ],
)

# =========================
# Stream Event Analyzer Class
# =========================

class StreamEventAnalyzer:
    """
    Analyzes streaming events for statistics and text collection.
    """
    def __init__(self):
        self.event_counts: dict[str, str] = {}
        self.text_chunks: list[str] = []
        self.start_time: float = time.time()
        self.total_events: int = 0

    def process_event(self, event: StreamEvent) -> None:
        self.total_events += 1
        event_type: str = type(event).__name__
        self.event_counts[event_type] = self.event_counts.get(event_type, 0) + 1

        if event.type == "raw_response_event" and isinstance(
            event.data, ResponseTextDeltaEvent
        ):
            self.text_chunks.append(event.data.delta)

    def get_statistics(self) -> dict[str, any]:
        elapsed_time = time.time() - self.start_time
        return {
            "total_events": self.total_events,
            "event_types": self.event_counts,
            "elapsed_time": elapsed_time,
            "events_per_second": (
                self.total_events / elapsed_time if elapsed_time > 0 else 0
            ),
            "text_chunks_count": len(self.text_chunks),
            "total_text_length": sum(len(chunk) for chunk in self.text_chunks),
        }

# =========================
# Streaming Analysis Function
# =========================

async def stream_with_analysis(
    agent: Agent, user_input: str, scenario_name: str
) -> None:
    """
    Streams agent output, processes events in real-time, and prints analysis and statistics.
    Shows streamed text, event types, and verifies text consistency.
    """
    print("\n====================================")
    print(f"SCENARIO: {scenario_name}")
    print(f"USER INPUT: {user_input}")
    print("====================================")
    run_result_stream: RunResultStreaming = Runner.run_streamed(
        starting_agent=agent, input=user_input
    )

    collected_text: str = ""
    analyzer = StreamEventAnalyzer()

    print("✅ Streaming has started. Stream object created.")

    async for event in run_result_stream.stream_events():
        analyzer.process_event(event)
        event_type = type(event).__name__
        print(f"📡 Received event type: {event_type}")

        if event.type == "raw_response_event" and isinstance(
            event.data, ResponseTextDeltaEvent
        ):
            print(f"[TEXT] {event.data.delta}", end="", flush=True)
            collected_text += event.data.delta

        elif event.type == "run_item_stream_event":
            print(f"[RUN ITEM] {event.item.type}: {str(event.item)[:60]}...")

        else:
            print(f"[EVENT] {event_type}")

    print("\n--- Streaming Statistics ---")
    stats = analyzer.get_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")

    # Compare final output with streamed chunks
    if collected_text.strip() != run_result_stream.final_output.strip():
        print("\n⚠️  Text mismatch detected!")
        print(f"Streamed: '{collected_text.strip()[:100]}...'")
        print(f"Final:    '{run_result_stream.final_output.strip()[:100]}...'")
    else:
        print("\n✅ Text consistency verified")

# =========================
# Demo Function
# =========================

async def demonstrate_streaming_patterns():
    """
    Runs several advanced streaming scenarios with tools, handoffs, and agent transitions.
    Prints detailed analysis and statistics for each scenario.
    """
    scenarios: list[tuple[str, str]] = [
        ("Math Scenario", "Add 7 and 8"),
        ("Joke Scenario", "Tell me some jokes"),
        ("Complex Mix", "Add 3 and 5, then tell me jokes"),
        ("Unknown Tool", "Use the tool 'mind_reader' to analyze this"),
    ]
    for scenario_name, sample_query in scenarios:
        await stream_with_analysis(coordinator_agent, sample_query, scenario_name)
        await asyncio.sleep(1)

# =========================
# Main Entry Point
# =========================

async def main():
    """
    Main entry point for the advanced streaming demo suite. Runs all scenarios.
    """
    await demonstrate_streaming_patterns()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\n❌ Exception occurred: {e}")
        print(f"Error type: {type(e).__name__}")
