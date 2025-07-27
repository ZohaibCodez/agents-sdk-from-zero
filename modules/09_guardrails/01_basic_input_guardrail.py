"""
modules/09_guardrails/01_basic_input_guardrail.py

Basic Input Guardrail Demonstration

Description:
    Demonstrates how to use input guardrails in the OpenAI Agents SDK. Implements a movie-topic guardrail that only allows movie-related questions, showing how to catch and report guardrail tripwires in a robust, educational, and production-ready way.

Features:
    - Demonstrates robust input guardrail usage and tripwire handling
    - Visually distinct, clearly labeled output with banners and emoji
    - Educational comments and section headers
    - Explains intentional guardrail tripwire for demonstration

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
from typing import List

from agents import (
    Agent,
    InputGuardrailTripwireTriggered,
    OpenAIChatCompletionsModel,
    RunContextWrapper,
    Runner,
    TResponseInputItem,
    handoff,
    function_tool,
    input_guardrail,
)
from agents.guardrail import GuardrailFunctionOutput
from agents.lifecycle import RunHooks
from dotenv import find_dotenv, load_dotenv
from openai import AsyncOpenAI

# =========================
# Environment & Model Setup
# =========================
load_dotenv(find_dotenv())

# Configuration constants
GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")
GEMINI_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_MODEL_NAME: str = "gemini-2.0-flash"

# Validate API key
if not GEMINI_API_KEY:
    print("❌ GEMINI_API_KEY environment variable is required but not found.")
    raise ValueError("GEMINI_API_KEY environment variable is required but not found.")

print(f"\n==============================")
print(f"🚀 Initializing Gemini client with model: {GEMINI_MODEL_NAME}")
print(f"==============================\n")

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


@input_guardrail
async def movie_topic_guardrail(
    ctx: RunContextWrapper, agent: Agent, input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    """
    Guardrail that only allows movie-related questions.
    Triggers a tripwire if the input is not movie-related (intentional for demonstration).
    """
    if isinstance(input, str):
        user_message = input
    else:
        user_message = str(input[-1]) if input else ""

    movie_keywords = [
        "movie",
        "film",
        "actor",
        "actress",
        "director",
        "cinema",
        "hollywood",
    ]

    if not any(keyword.lower() in user_message.lower() for keyword in movie_keywords):
        print(f"[GUARDRAIL] Input Guardrail Triggered. Not movie related: {user_message}")
        return GuardrailFunctionOutput(
            output_info={
                "reason": "Not movie related",
                "original_message": user_message,
            },
            tripwire_triggered=True,
        )

    print("[GUARDRAIL] Input Guardrail Passed Successfully. No tripwire triggered.")
    return GuardrailFunctionOutput(
        output_info={"status": "movie related", "message_length": len(user_message)},
        tripwire_triggered=False,
    )


async def demo_movie_topic_guardrail_only():
    """
    Test agent with only the movie topic guardrail active.
    Demonstrates both passing and tripwire-triggering cases.
    """
    print("\n==============================")
    print("🎬 [DEMO] Movie Topic Guardrail Only")
    print("==============================\n")

    agent = Agent(
        name="FriendlyMovieBot",
        instructions="You only answer questions related to movies.",
        input_guardrails=[movie_topic_guardrail],  # Only this guardrail!
        model=model
    )

    test_cases = [
        "Tell me about the new movie releases.",  # Should pass
        "What is your name?",                    # Should trigger
        "Who is your favorite actor?",           # Should pass
        "Let's talk about food.",                # Should trigger
    ]

    for i, test_input in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i}: {test_input} ---")
        try:
            result = await Runner.run(agent, test_input, max_turns=1)
            print(f"🤖 [OUTPUT] Agent response: {result.final_output}")

            if result.input_guardrail_results:
                for guardrail_result in result.input_guardrail_results:
                    print(
                        f"[GUARDRAIL] '{guardrail_result.guardrail.get_name()}': {guardrail_result.output.output_info}"
                    )
            else:
                print("[GUARDRAIL] No input guardrail results recorded.")

        except InputGuardrailTripwireTriggered as e:
            print(f"❗ [TRIPWIRE] Input guardrail tripwire triggered: {e}")
            print(f"[GUARDRAIL] Triggered by guardrail result: {e.guardrail_result}")

        except Exception as e:
            print(f"❌ [ERROR] Error in test case {i}: {e}")


async def main():
    """
    Run the movie topic guardrail demonstration.
    """
    print("\n==================================================")
    print("🎬 Running Movie Topic Guardrail Only Demo")
    print("==================================================\n")
    await demo_movie_topic_guardrail_only()
    print("\n==================================================")
    print("✅ [COMPLETE] Movie Topic Guardrail demo finished!")
    print("==================================================\n")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\n❌ [ERROR] Unhandled exception in main: {e}\n")
