"""
Error Handling Tools Demo - OpenAI Agents SDK

This module demonstrates advanced error handling in function tools with the OpenAI Agents SDK.
It showcases:
- Custom error handlers for function tools
- Different error handling strategies (custom vs strict)
- Streaming execution with error events
- Tool failure recovery and user guidance
- Error type detection and appropriate responses
- Graceful degradation when tools fail

The demo implements various mathematical operations with different error handling approaches.

Environment variables:
    - MISTRAL_API_KEY: Required for OpenRouter API access.

Author: Zohaib Khan
"""

import asyncio
import os
import random
from typing import Any
from dotenv import load_dotenv, find_dotenv

from agents import (
    Agent,
    Runner,
    set_tracing_disabled,
    function_tool,
    RunContextWrapper,
    ItemHelpers,
)
from agents.extensions.models.litellm_model import LitellmModel

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

print(f"🚀 Initializing Litellm client with model: {MODEL_NAME}")

model = LitellmModel(model=MODEL_NAME, api_key=MISTRAL_API_KEY)
print(f"✅ Model configured successfully\n")


# =========================
# Function Tools
# =========================


@function_tool
def divide_numbers(a: float, b: float) -> str:
    """
    Divide two numbers with basic error handling.

    Args:
        a: Dividend
        b: Divisor

    Returns:
        str: Division result or error message

    Raises:
        ValueError: When attempting to divide by zero
    """
    if b == 0:
        raise ValueError("Cannot divide by zero!")
    return f"{a} ÷ {b} = {a / b}"


@function_tool
def random_error_function() -> str:
    """
    Function that randomly fails to demonstrate error handling.

    Returns:
        str: Success message or raises RuntimeError randomly

    Raises:
        RuntimeError: Randomly with 50% probability
    """
    if random.random() < 0.5:
        raise RuntimeError("Random error occurred!")
    return "Function executed successfully!"


def custom_error_handler(ctx: RunContextWrapper[Any], error: Exception) -> str:
    """
    Custom error handler for function tools.

    Args:
        ctx: RunContextWrapper for managing execution context
        error: The exception that occurred

    Returns:
        str: User-friendly error message based on error type
    """
    if isinstance(error, ValueError):
        return f"Math error: {str(error)} Please check your input values."
    elif isinstance(error, RuntimeError):
        return f"Runtime issue: {str(error)} You might want to try again."
    else:
        return (
            f"Unexpected error: {str(error)} Please contact support if this persists."
        )


@function_tool(failure_error_function=custom_error_handler)
def protected_divide(a: float, b: float) -> str:
    """
    Protected division with custom error handling.

    Args:
        a: Dividend
        b: Divisor

    Returns:
        str: Division result or custom error message
    """
    if b == 0:
        raise ValueError("Division by zero is not allowed")
    return f"Protected division result: {a} ÷ {b} = {a / b}"


@function_tool(failure_error_function=None)
def strict_divide(a: float, b: float) -> str:
    """
    Strict division that allows errors to propagate.

    Args:
        a: Dividend
        b: Divisor

    Returns:
        str: Division result

    Raises:
        ValueError: When attempting to divide by zero
    """
    if b == 0:
        raise ValueError("Division by zero is not allowed")
    return f"Strict division result: {a} ÷ {b} = {a / b}"


# =========================
# Agent Configuration
# =========================

agent = Agent(
    name="Math Assistant",
    instructions="""You are a math assistant that can perform various calculations.
    When errors occur, try to understand what went wrong and provide helpful guidance to the user.
    If a calculation fails, suggest alternative approaches or corrected inputs.But first for every math related query you have to first use math tools like for division query always first use divide_numbers tool""",
    tools=[divide_numbers, random_error_function, protected_divide, strict_divide],
    model=model,
)


# =========================
# Streaming Runner Wrapper
# =========================


async def run_and_debug_full(agent, input: str):
    """
    Execute agent with streaming and detailed event debugging.

    Args:
        agent: The agent to execute
        input: User input string
    """
    print(f"\n🧪 Input: {input}")
    result = Runner.run_streamed(agent, input=input)

    current_agent = agent.name

    async for event in result.stream_events():
        if event.type == "raw_response_event":
            continue
        elif event.type == "agent_updated_stream_event":
            current_agent = event.new_agent.name
            print(f"🔄 Agent switched to: {current_agent}")
        elif event.type == "run_item_stream_event":
            item = event.item
            if item.type == "handoff_call_item":
                print("🤝 Handing off...")
            elif item.type == "handoff_output_item":
                print(
                    f"📤 Switching from {item.source_agent.name} to {item.target_agent.name}"
                )
            elif item.type == "tool_call_item":
                tool_name = getattr(item.raw_item, "name", "Unknown Tool")
                print(f"🔧 Tool called: {tool_name}")
            elif item.type == "tool_call_output_item":
                print(f"📊 Tool Output: {item.output}")
            elif item.type == "message_output_item":
                message_text = ItemHelpers.text_message_output(item)
                print(f"💬 {current_agent} says: {message_text}")

    print(f"Final Output: {result.final_output}")


# =========================
# Main Execution
# =========================


async def main():
    """
    Execute the error handling tools demo.

    Demonstrates various error handling strategies with mathematical operations
    and streaming execution with detailed event monitoring.
    """
    print("🔧 Error Handling Tools Demo")
    print("=" * 50)

    test_cases = [
        "Divide 10 by 2",
        "Divide 10 by 0",
        "Run the random error function",
        "Use protected_divide to divide 15 by 3",
        "Use protected_divide to divide 15 by 0",
        "Use strict_divide to divide 12 by 0",
        "Calculate these divisions: 100÷5, 50÷10, and 25÷0. For any that fail, explain what went wrong.",
    ]

    try:
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n{'='*20} Test Case {i} {'='*20}")
            try:
                await run_and_debug_full(agent, test_case)
            except Exception as e:
                print(f"❌ Exception occurred: {type(e).__name__} → {e}")
                print("=" * 50)

    except Exception as e:
        print(f"❌ Error during execution: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
