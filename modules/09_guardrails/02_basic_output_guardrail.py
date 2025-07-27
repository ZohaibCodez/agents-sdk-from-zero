"""
modules/09_guardrails/02_basic_output_guardrail.py

Basic Output Guardrail Demonstration

Description:
    Demonstrates how to use output guardrails in the OpenAI Agents SDK. Implements a response safety guardrail that blocks responses containing sensitive words, showing how to catch and report guardrail tripwires in a robust, educational, and production-ready way.

Features:
    - Demonstrates robust output guardrail usage and tripwire handling
    - Visually distinct, clearly labeled output with banners and emoji
    - Educational comments and section headers
    - Explains intentional output guardrail tripwire for demonstration

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

from agents import (
    Agent,
    OpenAIChatCompletionsModel,
    OutputGuardrailTripwireTriggered,
    RunContextWrapper,
    Runner,
    TResponseInputItem,
    handoff,
    function_tool,
    input_guardrail,
    output_guardrail,
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


@output_guardrail
def response_safety_guardrail(
    context: RunContextWrapper, agent: Agent, agent_output
) -> GuardrailFunctionOutput:
    """
    Output guardrail that blocks responses containing sensitive words.
    Triggers a tripwire if the response contains any word from a sensitive list (intentional for demonstration).
    """
    response_text = str(agent_output) if agent_output else ""

    # Example sensitive words list:
    sensitive_words = ["password", "secret", "confidential"]

    found_sensitive = [
        word for word in sensitive_words if word.lower() in response_text.lower()
    ]

    if found_sensitive:
        print(f"🚫 Output blocked by response_safety_guardrail: Found sensitive words: {found_sensitive}")
        return GuardrailFunctionOutput(
            output_info={
                "triggered_by": "response_safety_guardrail",
                "found_sensitive_words": found_sensitive,
                "agent_name": agent.name,
            },
            tripwire_triggered=True,
        )

    print(f"✅ Output allowed by response_safety_guardrail: No unsafe patterns found")
    return GuardrailFunctionOutput(
        output_info={
            "triggered_by": "response_safety_guardrail",
            "status": "safe",
            "agent_name": agent.name,
        },
        tripwire_triggered=False,
    )


async def demo_response_safety_guardrail_only():
    """
    Test agent with only response_safety_guardrail active.
    Demonstrates both passing and tripwire-triggering cases.
    """
    print("\n==============================")
    print("🛡️ [DEMO] Response Safety Output Guardrail Only")
    print("==============================\n")

    agent = Agent(
        name="SafetyTestAgent",
        instructions="Sometimes I might mention secrets or passwords in responses.",
        output_guardrails=[response_safety_guardrail],
        model=model,
    )

    test_cases = [
        "Here is your password: 12345",  # Should trigger
        "This is just a normal message.",  # Should pass
        "Keep this confidential please.",  # Should trigger
    ]

    for i, test_input in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i}: {test_input} ---")
        try:
            result = await Runner.run(agent, test_input, max_turns=1)

            if result.output_guardrail_results:
                for guardrail_result in result.output_guardrail_results:
                    status = (
                        "✅ ALLOWED"
                        if not guardrail_result.output.tripwire_triggered
                        else "❌ BLOCKED"
                    )
                    print(
                        f"[{status}] {guardrail_result.guardrail.get_name()}: {guardrail_result.output.output_info}"
                    )
            else:
                print("✅ No output guardrail triggered — response allowed.")

        except OutputGuardrailTripwireTriggered as e:
            print(f"❌ Output blocked by: {e.guardrail_result.guardrail.get_name()}")
            print(f"Reason: {e.guardrail_result.output.output_info}")

        except Exception as e:
            print(f"⚠️ [ERROR] Error in test case {i}: {e}")


async def main():
    """
    Run the response safety output guardrail demonstration.
    """
    print("\n==================================================")
    print("🛡️ Running Response Safety Output Guardrail Only Demo")
    print("==================================================\n")
    await demo_response_safety_guardrail_only()
    print("\n==================================================")
    print("✅ [COMPLETE] Response Safety Output Guardrail demo finished!")
    print("==================================================\n")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\n❌ [ERROR] Unhandled exception in main: {e}\n")
