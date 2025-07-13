"""
07_tool_choice_streaming.py

Tool Choice and Streaming Example
https://openai.github.io/openai-agents-python/tools/

Demonstrates advanced tool choice configurations and streaming with function tools.

Features demonstrated:
- ModelSettings with different tool_choice options (auto, required, none, specific)
- Parallel vs sequential tool calls
- Streaming tool execution with event handling
- Study productivity tools with focus scoring

Environment variables:
    - MISTRAL_API_KEY: Required for Litellm API access.

Author: Zohaib Khan
"""

import asyncio
import os
from typing import Any
from agents.extensions.models.litellm_model import LitellmModel
from dotenv import load_dotenv, find_dotenv

from openai import AsyncOpenAI
from agents.handoffs import Handoff, handoff
from openai.resources.models import Models
from openai.types.responses import ResponseTextDeltaEvent
from agents import (
    Agent,
    RunResult,
    Runner,
    OpenAIChatCompletionsModel,
    RunResultStreaming,
    model_settings,
    set_tracing_disabled,
    function_tool,
    RunContextWrapper,
    ItemHelpers,
    ModelSettings,
)
from dataclasses import dataclass

# =========================
# Environment & Model Setup
# =========================


load_dotenv(find_dotenv())
set_tracing_disabled(True)

# Configuration constants
MISTRAL_API_KEY: str | None = os.getenv("MISTRAL_API_KEY")
MODEL_NAME: str = "mistral/mistral-small-2506"

# Validate API key
if not MISTRAL_API_KEY:
    print("❌ MISTRAL_API_KEY environment variable is required but not found.")
    raise ValueError("MISTRAL_API_KEY environment variable is required but not found.")

print(f"🚀 Initializing OpenRouter client with model: {MODEL_NAME}")

model = LitellmModel(model=MODEL_NAME, api_key=MISTRAL_API_KEY)
print(f"✅ Model configured successfully\n")

# =========================
# Function Tools
# =========================


@function_tool
def study_tip(subject: str) -> str:
    """
    Provide study tips for a specific subject.

    Args:
        subject: The subject to provide study tips for

    Returns:
        A study tip for the specified subject
    """
    try:
        return f"For {subject}, use spaced repetition and active recall to study effectively."
    except Exception as e:
        return f"❌ Error providing study tip: {str(e)}"


@function_tool
def get_focus_score(hours_slept: int, distractions: int) -> str:
    """
    Calculate focus score based on sleep and distractions.

    Args:
        hours_slept: Number of hours slept
        distractions: Number of distractions encountered

    Returns:
        A focus score out of 100
    """
    try:
        score = max(0, min(100, (hours_slept * 10) - (distractions * 5)))
        return f"Your estimated focus score is {score}/100."
    except Exception as e:
        return f"❌ Error calculating focus score: {str(e)}"


@function_tool
def suggest_break(activity: str = "walking") -> str:
    """
    Suggest a break activity for mental refreshment.

    Args:
        activity: The type of break activity (default: walking)

    Returns:
        A break suggestion
    """
    try:
        return f"Take a 5-minute {activity} break to refresh your mind."
    except Exception as e:
        return f"❌ Error suggesting break: {str(e)}"


# =========================
# Agent Definition
# =========================

base_agent: Agent = Agent(
    name="StudyCoach",
    instructions="You help students improve their productivity using tools for tips, focus, and breaks.",
    tools=[study_tip, get_focus_score, suggest_break],
    model=model,
    model_settings=ModelSettings(tool_choice="auto"),
)

# =========================
# Main Execution
# =========================


async def main():
    """
    Demonstrate various tool choice configurations and streaming capabilities.

    Tests different ModelSettings configurations:
    - Auto tool choice (default behavior)
    - Required tool choice (force tool use)
    - No tool choice (forbid tool use)
    - Specific tool choice (force specific tool)
    - Reset behavior with tool choice
    - Parallel vs sequential tool calls
    - Streaming with event handling
    """
    try:
        print("\n" + "=" * 60)
        print("🔧 TOOL CHOICE & STREAMING DEMO")
        print("=" * 60)

        # 1. default behavior (tool_choice = "auto")
        print("\n" + "🔄 TEST 1: Auto Tool Choice\n" + "=" * 40)
        sample_query_1 = "Give me a tip for studying statistics"
        print(f"📥 INPUT: {sample_query_1}")
        print("-" * 60)
        result_1 = await Runner.run(base_agent, input=sample_query_1)
        print(f"📤 OUTPUT: {result_1.final_output}")
        print("✅ Test 1 completed successfully!")

        # 2. Force tool use (tool_choice = "required")
        print("\n" + "🔧 TEST 2: Required Tool Choice\n" + "=" * 35)
        agent_required: Agent = base_agent.clone(
            model_settings=ModelSettings(tool_choice="required")
        )
        sample_query_2 = "Hey, there."
        print(f"📥 INPUT: {sample_query_2}")
        print("-" * 60)
        result_2 = await Runner.run(agent_required, input=sample_query_2)
        print(f"📤 OUTPUT: {result_2.final_output}")
        print("✅ Test 2 completed successfully!")

        # 3. Forbid tool use (tool_choice = "none")
        print("\n" + "🚫 TEST 3: No Tool Choice\n" + "=" * 35)
        agent_no_tools: Agent = base_agent.clone(
            model_settings=ModelSettings(tool_choice="none")
        )
        sample_query_3 = "Give me a tip for studying statistics"
        print(f"📥 INPUT: {sample_query_3}")
        print("-" * 60)
        result_3 = await Runner.run(agent_no_tools, input=sample_query_3)
        print(f"📤 OUTPUT: {result_3.final_output}")
        print("✅ Test 3 completed successfully!")

        # 4. Force specific tool use (tool_choice = "tool_name")
        print("\n" + "🎯 TEST 4: Specific Tool Choice\n" + "=" * 35)
        agent_specific: Agent = base_agent.clone(
            model_settings=ModelSettings(tool_choice="suggest_break")
        )
        sample_query_4 = "Hey, there. I need some info"
        print(f"📥 INPUT: {sample_query_4}")
        print("-" * 60)
        result_4 = await Runner.run(agent_specific, input=sample_query_4)
        print(f"📤 OUTPUT: {result_4.final_output}")
        print("✅ Test 4 completed successfully!")

        # 5. Tool Choice with Reset Behavior (reset_tool_choice="False")
        print("\n" + "🔄 TEST 5: Reset Tool Choice Behavior\n" + "=" * 30)
        agent_reset_false: Agent = base_agent.clone(
            model_settings=ModelSettings(tool_choice="required"),
            reset_tool_choice=False,
        )
        sample_query_5 = "Hey, there. Give me a tip for studying statistics. I need a focus score and also a break suggestion"
        print(f"📥 INPUT: {sample_query_5}")
        print("-" * 60)
        try:
            result_5: RunResult = await Runner.run(
                agent_reset_false, input=sample_query_5
            )
            print(f"📤 OUTPUT: {result_5.final_output}")
            print("✅ Test 5 completed successfully!")
        except Exception as e:
            print(f"❌ Error: {type(e).__name__}: {str(e)}")
            print("⚠️ Test 5 failed as expected (reset behavior)")

        # 6. Parallel Tool Calls (parallel_tool_calls=True)
        print("\n" + "⚡ TEST 6: Parallel Tool Calls\n" + "=" * 35)
        agent_parallel: Agent = base_agent.clone(
            model_settings=ModelSettings(tool_choice="auto", parallel_tool_calls=True)
        )
        sample_query_6 = "I slept 7 hours and had 2 distractions. Also suggest a break."
        print(f"📥 INPUT: {sample_query_6}")
        print("-" * 60)
        try:
            result_6: RunResultStreaming = Runner.run_streamed(
                agent_parallel, input=sample_query_6
            )
            print("📡 STREAMING EVENTS:")
            async for event in result_6.stream_events():
                if event.type == "raw_response_event":
                    continue
                elif event.type == "agent_updated_stream_event":
                    print(f"🤖 Agent updated: {event.new_agent.name}")
                elif event.type == "run_item_stream_event":
                    item = event.item
                    if item.type == "tool_call_item":
                        print("🔧 Tool called")
                    elif item.type == "tool_call_output_item":
                        print(f"📤 Tool output: {item.output}")
            print(f"📤 OUTPUT: {result_6.final_output}")
            print("✅ Test 6 completed successfully!")
        except Exception as e:
            print(f"❌ Error: {type(e).__name__}: {str(e)}")
            print("⚠️ Test 6 failed (parallel tool calls not supported)")

        # 7. Sequential Tool Calls (parallel_tool_calls=False) - Default
        print("\n" + "📡 TEST 7: Sequential Tool Calls with Streaming\n" + "=" * 25)
        agent_sequential: Agent = base_agent.clone(
            model_settings=ModelSettings(parallel_tool_calls=False)  # by default
        )
        sample_query_7: str = (
            "Give me a study tip for math and calculate my focus score: 6 hours sleep, 3 distractions"
        )
        print(f"📥 INPUT: {sample_query_7}")
        print("-" * 60)

        result_7: RunResultStreaming = Runner.run_streamed(
            agent_sequential, input=sample_query_7
        )
        print("📡 STREAMING EVENTS:")
        async for event in result_7.stream_events():
            if event.type == "raw_response_event":
                continue
            elif event.type == "agent_updated_stream_event":
                print(f"🤖 Agent updated: {event.new_agent.name}")
            elif event.type == "run_item_stream_event":
                item = event.item
                if item.type == "tool_call_item":
                    print("🔧 Tool called")
                elif item.type == "tool_call_output_item":
                    print(f"📤 Tool output: {item.output}")

        print(f"\n📤 FINAL OUTPUT: {result_7.final_output}")
        print("✅ Test 7 completed successfully!")

        print("\n" + "=" * 60)
        print("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)

    except Exception as e:
        print(f"❌ Error during execution: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
