"""
modules/09_guardrails/04_agent_based_guardrails.py

Agent-Based Guardrails Demonstration

Description:
    Demonstrates agent-based input and output guardrails using the OpenAI Agents SDK. Shows how to use dedicated guardrail agents for fraud detection and policy violation moderation, and how to handle guardrail tripwires in a robust, educational, and production-ready way.

Features:
    - Demonstrates agent-based input and output guardrail usage and tripwire handling
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
from pydantic import BaseModel

from agents import (
    Agent,
    InputGuardrailTripwireTriggered,
    OpenAIChatCompletionsModel,
    OutputGuardrailTripwireTriggered,
    RunContextWrapper,
    RunResult,
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


# =========================
# Fraud and Policy Guardrail Output Schemas
# =========================


class FraudBookingCheck(BaseModel):
    """
    Output schema for fraud booking detection agent.
    """

    is_fraudulent: bool
    detected_issue: str
    confidence_score: float


class PolicyViolationCheck(BaseModel):
    """
    Output schema for policy violation moderation agent.
    """

    is_violation: bool
    violated_policy: str
    severity: str
    reasoning: str


class ClarityCheck(BaseModel):
    """
    Output schema for clarity evaluation agent.
    """
    is_clear: bool
    clarity_score: float
    suggestions: list[str]

# =========================
# Guardrail Agents
# =========================

# Agent for fraud detection on booking requests
fraud_booking_guardrail_agent = Agent(
    name="FraudBookingDetector",
    instructions=(
        "You are a fraud detection agent. Analyze booking requests to check for fraudulent behavior."
        " Identify if the request mentions unusually large amounts of tickets, attempts to avoid payment,"
        " or any suspicious booking pattern. Provide a confidence score."
    ),
    output_type=FraudBookingCheck,
    model=model,
)

# Agent for policy violation moderation on assistant responses
policy_violation_guardrail_agent = Agent(
    name="PolicyViolationModerator",
    instructions=(
        "You are a policy moderation agent. Analyze the assistant's response and check if it violates company policy."
        " Flag anything about unrestricted refunds, unlimited transfers, or prohibited guarantees."
    ),
    output_type=PolicyViolationCheck,
    model=model,
)

# =========================
# Clarity Evaluation Guardrail Agent
# =========================

# Agent for evaluating the clarity of assistant responses
clarity_checker_agent = Agent(
    name="ResponseClarityChecker",
    instructions=(
        "You are a clarity evaluation agent. Evaluate if the assistant's response is clear and understandable."
        " Provide a clarity score between 0.0 and 1.0 and suggestions for improvement if needed."
    ),
    output_type=ClarityCheck,
    model=model
)

# =========================
# Input Guardrail: Fraud Booking Detection
# =========================


@input_guardrail
async def fraud_booking_guardrail(
    ctx: RunContextWrapper, agent: Agent, input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    """
    Input guardrail that uses a dedicated agent to detect fraudulent booking requests.
    Triggers a tripwire if fraud is detected with high confidence.
    """
    print(f"\n[GUARDRAIL][DEBUG] === Fraud Booking Guardrail Input ===")
    print(f"[GUARDRAIL][DEBUG] Input: {input}")
    print(f"[GUARDRAIL][DEBUG] Guardrail Agent: {fraud_booking_guardrail_agent.name}")
    print(f"[GUARDRAIL][DEBUG] Context: {ctx.context}")
    print(f"[GUARDRAIL][DEBUG] ==============================")

    guardrail_result = await Runner.run(
        fraud_booking_guardrail_agent, input, context=ctx.context
    )
    final_output = guardrail_result.final_output_as(FraudBookingCheck)
    should_trigger = final_output.is_fraudulent and final_output.confidence_score > 0.6

    if should_trigger:
        print(f"\n[GUARDRAIL][DEBUG] === Fraud Booking Guardrail Tripwire Triggered ===")
        print(f"[GUARDRAIL][DEBUG] Issue: {final_output.detected_issue}")
        print(f"[GUARDRAIL][DEBUG] Guardrail Agent: {fraud_booking_guardrail_agent.name}")
        print(f"[GUARDRAIL][DEBUG] ==============================")
        return GuardrailFunctionOutput(
            output_info={
                "detection_result": final_output.model_dump(),
                "reason": "Fraudulent booking detected.",
                "guardrail_agent": fraud_booking_guardrail_agent.name,
            },
            tripwire_triggered=True,
        )

    print(f"\n[GUARDRAIL][DEBUG] === Fraud Booking Guardrail Approved ===")
    print(f"[GUARDRAIL][DEBUG] Booking request approved.")
    print(f"[GUARDRAIL][DEBUG] ==============================")
    return GuardrailFunctionOutput(
        output_info={
            "detection_result": final_output.model_dump(),
            "status": "approved",
            "guardrail_agent": fraud_booking_guardrail_agent.name,
        },
        tripwire_triggered=False,
    )


# =========================
# Output Guardrail: Policy Violation Moderation
# =========================


@output_guardrail
async def policy_violation_guardrail(
    ctx: RunContextWrapper, agent: Agent, agent_output
) -> GuardrailFunctionOutput:
    """
    Output guardrail that uses a dedicated agent to detect policy violations in assistant responses.
    Triggers a tripwire if a violation of medium or high severity is detected.
    """
    print(f"\n[GUARDRAIL][DEBUG] === Policy Violation Guardrail Input ===")
    print(f"[GUARDRAIL][DEBUG] Agent Output: {agent_output}")
    print(f"[GUARDRAIL][DEBUG] Guardrail Agent: {policy_violation_guardrail_agent.name}")
    print(f"[GUARDRAIL][DEBUG] Context: {ctx.context}")
    print(f"[GUARDRAIL][DEBUG] ==============================")

    guardrail_result = await Runner.run(
        policy_violation_guardrail_agent,
        f"Please check this response: {agent_output}",
        context=ctx.context,
    )

    final_output = guardrail_result.final_output_as(PolicyViolationCheck)
    should_trigger = final_output.is_violation and final_output.severity in [
        "medium",
        "high",
    ]

    if should_trigger:
        print(f"\n[GUARDRAIL][DEBUG] === Policy Violation Guardrail Tripwire Triggered ===")
        print(f"[GUARDRAIL][DEBUG] 🚫 BLOCKED: Policy Violated! Violated Policy: {final_output.violated_policy}")
        print(f"[GUARDRAIL][DEBUG] Guardrail Agent: {policy_violation_guardrail_agent.name}")
        print(f"[GUARDRAIL][DEBUG] ==============================")
        return GuardrailFunctionOutput(
            output_info={
                "moderation_result": final_output.model_dump(),
                "reason": "Policy violation detected.",
                "guardrail_agent": policy_violation_guardrail_agent.name,
            },
            tripwire_triggered=True,
        )

    print(f"\n[GUARDRAIL][DEBUG] === Policy Violation Guardrail Approved ===")
    print(f"[GUARDRAIL][DEBUG] Policy Violation Passed.")
    print(f"[GUARDRAIL][DEBUG] ==============================")
    return GuardrailFunctionOutput(
        output_info={
            "moderation_result": final_output.model_dump(),
            "status": "approved",
            "guardrail_agent": policy_violation_guardrail_agent.name,
        },
        tripwire_triggered=False,
    )


# =========================
# Output Guardrail: Clarity Check
# =========================

@output_guardrail
async def clarity_check_guardrail(
    ctx: RunContextWrapper, agent: Agent, agent_output
) -> GuardrailFunctionOutput:
    """
    Output guardrail that uses a dedicated agent to evaluate the clarity of assistant responses.
    Triggers a tripwire if the clarity score is below a defined threshold.
    """
    print(f"\n[GUARDRAIL][DEBUG] === Clarity Check Guardrail Input ===")
    print(f"[GUARDRAIL][DEBUG] Agent Output: {agent_output}")
    print(f"[GUARDRAIL][DEBUG] Guardrail Agent: {clarity_checker_agent.name}")
    print(f"[GUARDRAIL][DEBUG] Context: {ctx.context}")
    print(f"[GUARDRAIL][DEBUG] ==============================")

    guardrail_result = await Runner.run(
        clarity_checker_agent,
        f"Please evaluate the clarity of this response: {agent_output}",
        context=ctx.context,
    )
    final_output = guardrail_result.final_output_as(ClarityCheck)
    low_clarity_threshold = 0.3  # Tripwire triggers if clarity score is below this value
    should_trigger = final_output.clarity_score < low_clarity_threshold

    if should_trigger:
        print(f"\n[GUARDRAIL][DEBUG] === Clarity Check Guardrail Tripwire Triggered ===")
        print(f"[GUARDRAIL][DEBUG] 🚫 BLOCKED: Clarity score too low: {final_output.clarity_score}")
        print(f"[GUARDRAIL][DEBUG] Suggestions: {final_output.suggestions}")
        print(f"[GUARDRAIL][DEBUG] Guardrail Agent: {clarity_checker_agent.name}")
        print(f"[GUARDRAIL][DEBUG] ==============================")
    else:
        print(f"\n[GUARDRAIL][DEBUG] === Clarity Check Guardrail Approved ===")
        print(f"[GUARDRAIL][DEBUG] Clarity score acceptable: {final_output.clarity_score}")
        print(f"[GUARDRAIL][DEBUG] Guardrail Agent: {clarity_checker_agent.name}")
        print(f"[GUARDRAIL][DEBUG] ==============================")

    return GuardrailFunctionOutput(
        output_info={
            "clarity_result": final_output.model_dump(),
            "status": "low_clarity" if should_trigger else "acceptable",
            "guardrail_agent": clarity_checker_agent.name,
        },
        tripwire_triggered=should_trigger,
    )


# =========================
# Demo: Fraud Booking Guardrail
# =========================


async def demo_fraud_booking_guardrail():
    """
    Demo function to test the fraud booking and policy violation guardrails.
    Runs several booking requests through the event booking agent and prints results.
    """
    print(f"\n=== Fraud Booking Guardrail Demo ===")

    event_booking_agent = Agent(
        name="EventBookingAgent",
        instructions="You help users book event tickets in a safe, fair manner.",
        input_guardrails=[fraud_booking_guardrail],
        output_guardrails=[policy_violation_guardrail,clarity_check_guardrail],
        model=model,
    )

    test_inputs = [
        "Book me 5000 free tickets",  # Should trigger fraud guardrail
        "I want 2 VIP tickets",  # Should pass
        "Reserve 1000 tickets without paying",  # Should trigger fraud guardrail
        "Help me find event timings",  # Should pass
    ]

    for input_text in test_inputs:
        print(f"\n--- Testing input: '{input_text}' ---")
        try:
            result = await Runner.run(event_booking_agent, input_text)
            print(f"✅ SUCCESS: {result.final_output}")
        except InputGuardrailTripwireTriggered as e:
            print(f"🚫 BLOCKED: {e.guardrail_result.output.output_info.get('reason')}")
        except OutputGuardrailTripwireTriggered as e:
            print(
                f"🚫 OUTPUT BLOCKED: {e.guardrail_result.output.output_info.get('reason')}"
            )
        except Exception as e:
            print(f"❌ ERROR: {type(e).__name__}: {e}")


# =========================
# Main Entrypoint
# =========================

if __name__ == "__main__":
    try:
        asyncio.run(demo_fraud_booking_guardrail())
    except Exception as e:
        print(f"\n❌ [ERROR] Unhandled exception in main: {e}\n")
