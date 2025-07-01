"""
09_runner_exceptions.py

Demonstrates exception handling using OpenAI Agents SDK Runner.

This module helps you understand how to handle different failure scenarios when running agents.

Covers:
- MaxTurnsExceeded
- UserError (e.g., misconfigured agent)
- InputGuardrailTripwireTriggered
- OutputGuardrailTripwireTriggered
- ModelBehaviorError (concept only)

Concepts learned here are foundational for building robust agents that don't break unexpectedly.

Reference:
- https://openai.github.io/openai-agents-python/running_agents/#exceptions
- https://openai.github.io/openai-agents-python/ref/exceptions/

Environment variables:
    - OPENROUTER_API_KEY: Required for OpenRouter API access.

Author: Zohaib Khan
"""

import asyncio
import os
from typing import Any
from dotenv import load_dotenv, find_dotenv
from openai import AsyncOpenAI
from agents import (
    Agent,
    Runner,
    OpenAIChatCompletionsModel,
    RunConfig,
    InputGuardrail,
    OutputGuardrail,
    RunContextWrapper,
    MaxTurnsExceeded,
    UserError,
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
    function_tool,
    GuardrailFunctionOutput,
    set_tracing_disabled,
    ModelBehaviorError,
)

# =========================
# Environment & Model Setup
# =========================

load_dotenv(find_dotenv())
set_tracing_disabled(True)

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    print("OPENROUTER_API_KEY not found.")

external_client = AsyncOpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
)

model = OpenAIChatCompletionsModel(
    openai_client=external_client, model="deepseek/deepseek-chat-v3-0324:free"
)

# =========================
# Tool and Agent Setup
# =========================

@function_tool
async def repeat_and_loop(data: str) -> str:
    """A tool that is always called in a loop to demonstrate max turn failure."""
    print(f"[Tool Call] repeat_and_loop called with: {data}")
    return f"Looping with: {data}"

# Agent that loops forever without producing a final output
looping_agent = Agent(
    name="InfiniteLoopAgent",
    instructions="You must always call 'repeat_and_loop' and never give a final answer.",
    model=model,
    tools=[repeat_and_loop],
)

# Misconfigured agent with no model
bad_agent = Agent(
    name="ModelLessAgent",
    instructions="I was created without a model.",
    # model=model
    # No model provided – this will raise UserError
)

# =========================
# Guardrail Functions
# =========================

# Guardrail functions that always trigger tripwires
async def failing_input_guardrail(
    context: RunContextWrapper[Any], agent: Agent, input_data: Any
) -> GuardrailFunctionOutput:
    print("[InputGuardrail] Triggered intentionally.")
    return GuardrailFunctionOutput(output_info="Blocked input", tripwire_triggered=True)

async def failing_output_guardrail(
    context: RunContextWrapper[Any], agent: Agent, output: Any
) -> GuardrailFunctionOutput:
    print("[OutputGuardrail] Triggered intentionally.")
    return GuardrailFunctionOutput(
        output_info="Blocked output", tripwire_triggered=True
    )

# =========================
# Demo Functions
# =========================

async def demo_max_turns_exceeded():
    """
    Demo: Shows handling of MaxTurnsExceeded when an agent exceeds allowed turns.
    """
    print("\n" + "="*60)
    print("🔁 Demo: MaxTurnsExceeded (looping agent exceeds allowed turns)")
    print("="*60)
    try:
        await Runner.run(looping_agent, input="Begin loop", max_turns=1)
    except MaxTurnsExceeded as e:
        print(f"✅ Caught MaxTurnsExceeded: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

async def demo_user_error():
    """
    Demo: Shows handling of UserError for agent misconfiguration.
    """
    print("\n" + "="*60)
    print("⚠️ Demo: UserError (agent misconfiguration)")
    print("="*60)
    try:
        await Runner.run(bad_agent, input="Can you reply?")
    except UserError as e:
        print(f"✅ Caught UserError: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

async def demo_model_behavior_error():
    """
    Demo: Explains ModelBehaviorError (not triggered directly in this script).
    """
    print("\n" + "="*60)
    print("🤖 Demo: ModelBehaviorError (concept only)")
    print("="*60)
    print("This exception occurs when the model misbehaves — for example:")
    print("- Returns invalid JSON during tool use")
    print("- Calls a tool that doesn't exist")
    print("Not triggered directly in this script, but useful to know for debugging.")

async def demo_guardrail_exceptions():
    """
    Demo: Shows handling of InputGuardrailTripwireTriggered and OutputGuardrailTripwireTriggered.
    """
    print("\n" + "="*60)
    print("🛡️ Demo: Guardrail Tripwire Exceptions")
    print("="*60)

    # Input Guardrail
    try:
        print("\n⛔ Triggering InputGuardrailTripwire...")
        input_fail_config = RunConfig(
            input_guardrails=[InputGuardrail(failing_input_guardrail)]
        )
        agent = Agent("InputTestAgent", model=model)
        await Runner.run(
            agent, input="This should be blocked", run_config=input_fail_config
        )
    except InputGuardrailTripwireTriggered as e:
        print(f"✅ Caught InputGuardrailTripwireTriggered: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

    # Output Guardrail
    try:
        print("\n⛔ Triggering OutputGuardrailTripwire...")
        output_fail_config = RunConfig(
            output_guardrails=[OutputGuardrail(failing_output_guardrail)]
        )
        agent = Agent("OutputTestAgent", model=model, instructions="Say anything.")
        await Runner.run(agent, input="Test output", run_config=output_fail_config)
    except OutputGuardrailTripwireTriggered as e:
        print(f"✅ Caught OutputGuardrailTripwireTriggered: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

# =========================
# Main Entry Point
# =========================

async def main():
    """
    Main entry point for the exception handling demo suite. Runs all demo cases.
    """
    print("\n🧪 Running 09_runner_exceptions.py (custom demo) ---")
    await demo_max_turns_exceeded()
    await demo_user_error()
    await demo_model_behavior_error()
    await demo_guardrail_exceptions()
    print("\n✅ Finished exception handling demo.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\n❌ Exception occurred: {e}")
        print(f"Error type: {type(e).__name__}")
