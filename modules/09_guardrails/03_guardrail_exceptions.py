"""
modules/09_guardrails/03_guardrail_exceptions.py

Comprehensive Guardrail Exception Handling Demonstration

Description:
    Demonstrates input and output guardrail exception handling using the OpenAI Agents SDK with Gemini API integration. Shows how to block sensitive topics in user input and disclaimers/unwanted phrases in agent output, and how to handle both InputGuardrailTripwireTriggered and OutputGuardrailTripwireTriggered exceptions in a robust, educational, and production-ready way.

Features:
    - Demonstrates robust input and output guardrail usage and tripwire handling
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

from agents import (
    Agent,
    InputGuardrailTripwireTriggered,
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

# ===============================
# Guardrail Function Definitions
# ===============================

@input_guardrail(name="strict_content_filter")
def strict_input_filter(
    ctx: RunContextWrapper, agent: Agent, input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    """
    Input Guardrail: Blocks sensitive topics in user input.
    """
    sensitive_topics: list[str] = ["finances", "medical", "legal", "personal"]
    user_message = input if isinstance(input, str) else str(input[-1]) if input else ""

    found_sensitive = [
        topic for topic in sensitive_topics if topic.lower() in user_message.lower()
    ]

    if found_sensitive:
        return GuardrailFunctionOutput(
            output_info={
                "trigerred_by": "strict_input_filter",
                "found_sensitive_topics": found_sensitive,
                "blocked_message": user_message,
            },
            tripwire_triggered=True,
        )

    return GuardrailFunctionOutput(
        output_info={"status": "approved", "message": user_message},
        tripwire_triggered=False,
    )


@output_guardrail(name="strict_output_filter")
def strict_output_filter(
    ctx: RunContextWrapper, agent: Agent, agent_output
) -> GuardrailFunctionOutput:
    """
    Output Guardrail: Blocks disclaimers or unwanted phrases in agent output.
    """
    forbidden_patterns = ["disclaimer", "not a substitute", "consult a professional"]
    response_text = str(agent_output) if agent_output else ""

    found_forbidden = [
        pattern
        for pattern in forbidden_patterns
        if pattern.lower() in response_text.lower()
    ]

    if found_forbidden:
        return GuardrailFunctionOutput(
            output_info={
                "triggered_by": "strict_output_filter",
                "forbidden_patterns_found": found_forbidden,
                "blocked_response": response_text,
            },
            tripwire_triggered=True,
        )

    return GuardrailFunctionOutput(
        output_info={"status": "approved", "response": response_text},
        tripwire_triggered=False,
    )

    
# =========================
# Test Scenarios (Clarified)
# =========================

async def demo_full_exception_handling():
    """
    ✅ Comprehensive Exception Handling Demo:
    
    Testing both input and output guardrails using 3 example queries:

    1️⃣ "Tell me about the weather." → Expected: Pass both input/output guardrails.
    2️⃣ "Help me with personal finances." → Expected: Blocked by input guardrail.
    3️⃣ "How to treat a headache at home?" → Expected: May be blocked by output guardrail if disclaimer appears.

    ✅ Behavior:
        - Input guardrail triggers BEFORE sending input to Gemini model.
        - Output guardrail triggers AFTER receiving response from Gemini model but BEFORE returning to user.
    """

    agent = Agent(
        name="FullyGuardedAgent",
        instructions="Be comprehensive. Always include disclaimers for medical advice.",
        input_guardrails=[strict_input_filter],
        output_guardrails=[strict_output_filter],
        model=model
    )

    test_inputs = [
        "Tell me about the weather.",                # ✔️ Normal
        "Help me with personal finances.",           # 🚫 Block by input guardrail
        "How to treat a headache at home?"           # 🚫 Possibly block by output guardrail
    ]

    for i, user_input in enumerate(test_inputs, 1):
        print(f"\n--- Full Test {i}: {user_input} ---")
        try:
            result = await Runner.run(agent, user_input)
            print(f"✅ Agent responded: {result.final_output}")

            # Clarified: Showing only if there were guardrail results
            if result.input_guardrail_results:
                for gr in result.input_guardrail_results:
                    print(f"[INFO] Input Guardrail: {gr.guardrail.get_name()} → {gr.output.output_info}")

            if result.output_guardrail_results:
                for gr in result.output_guardrail_results:
                    print(f"[INFO] Output Guardrail: {gr.guardrail.get_name()} → {gr.output.output_info}")

        except InputGuardrailTripwireTriggered as e:
            print(f"🚫 BLOCKED at INPUT by {e.guardrail_result.guardrail.get_name()}")
            print(f"Reason: {e.guardrail_result.output.output_info}")

        except OutputGuardrailTripwireTriggered as e:
            print(f"🚫 BLOCKED at OUTPUT by {e.guardrail_result.guardrail.get_name()}")
            print(f"Reason: {e.guardrail_result.output.output_info}")

        except Exception as e:
            print(f"⚠️ SYSTEM ERROR: {type(e).__name__}: {e}")

async def main():
    print("✅ Full Guardrail Exception Handling Test")
    await demo_full_exception_handling()

if __name__ == "__main__":
    asyncio.run(main())
