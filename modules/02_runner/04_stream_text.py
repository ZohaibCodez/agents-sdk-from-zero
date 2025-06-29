"""
04_streaming_joker_agent.py

🎭 Streaming Agent Runner Demonstration – "Joker" Agent

This module demonstrates how to use the OpenAI Agents SDK's streaming runner
to execute an agent that responds in real time using `Runner.run_streamed()`.

🧠 Concept Demonstrated:
- Running agents with streaming support
- Handling delta text via `ResponseTextDeltaEvent`
- Chaining multi-turn conversations using `to_input_list()`

🤖 Agent Role:
A humorous "Joker" who responds playfully to user input.

Author: Zohaib Khan
"""

import os
import asyncio
from dotenv import load_dotenv, find_dotenv
from typing import Optional, List, Dict, Any
from pprint import pprint

from agents import (
    Agent,
    Runner,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    set_tracing_disabled,
    RunResultStreaming,
)
from openai.types.responses import ResponseTextDeltaEvent

load_dotenv(find_dotenv())
set_tracing_disabled(True)

GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
GEMINI_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_MODEL_NAME: str = "gemini-2.0-flash"

if not GEMINI_API_KEY:
    raise ValueError(
        "❌ GEMINI_API_KEY environment variable is required but not found."
    )
print(f"🚀 Initializing Gemini model: {GEMINI_MODEL_NAME}")
external_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url=GEMINI_BASE_URL,
)

model = OpenAIChatCompletionsModel(
    openai_client=external_client, model=GEMINI_MODEL_NAME
)


async def run_joker_demo() -> None:
    """
    Main runner function that executes the Joker agent in two conversation turns:
    1. Initial greeting
    2. Follow-up request for five jokes

    Includes real-time delta output and conversation chaining using input lists.
    """
    joker: Agent = Agent(
        name="Joker",
        instructions="Act as a joker and give user queries answers like a joker.",
        model=model,
    )
    print("✅ Joker agent is ready.\n")

    # First user query
    user_input_1: str = "Hello Joker"
    print(f"🎯 User input: '{user_input_1}'\n")
    print("📡 Streaming Response (Turn 1):\n" + "-" * 60)

    run_result_1: RunResultStreaming = Runner.run_streamed(joker, user_input_1)
    async for event in run_result_1.stream_events():
        if event.type == "raw_response_event" and isinstance(
            event.data, ResponseTextDeltaEvent
        ):
            print(event.data.delta, end="", flush=True)

    print("\n" + "-" * 60)
    print("📋 Conversation After Turn 1:")
    pprint(run_result_1.to_input_list())

    # Create follow-up prompt
    user_input_2: Dict[str, str] = {"role": "user", "content": "Tell me five jokes"}
    full_convo: List[Dict[str, str]] = run_result_1.to_input_list() + [user_input_2]

    print("\n🎯 Follow-up query: 'Tell me five jokes'")
    print("📨 Full Conversation History Passed to Agent:")
    pprint(full_convo)

    print("\n📡 Streaming Response (Turn 2):\n" + "-" * 60)
    run_result_2: RunResultStreaming = Runner.run_streamed(joker, full_convo)
    async for event in run_result_2.stream_events():
        if event.type == "raw_response_event" and isinstance(
            event.data, ResponseTextDeltaEvent
        ):
            print(event.data.delta, end="", flush=True)

    print("\n" + "-" * 60)
    print("📋 Final Conversation After Turn 2:")
    pprint(run_result_2.to_input_list())

    print("\n✅ Joker demo completed successfully!\n")


if __name__ == "__main__":
    try:
        asyncio.run(run_joker_demo())
    except Exception as e:
        print(f"\n❌ Exception occurred: {e}")
