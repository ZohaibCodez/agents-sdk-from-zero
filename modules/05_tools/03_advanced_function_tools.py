"""
Advanced Function Tools Demo - OpenAI Agents SDK

This module demonstrates advanced function tool usage with the OpenAI Agents SDK.
It showcases:
- Creating custom FunctionTool instances with strict parameter validation
- Using Pydantic models with ConfigDict for extra field control
- Async tool functions with context management
- Tool introspection and metadata display
- Error handling and graceful failure
- Strict parameter validation with "forbid" extra fields

The demo implements a user data processing system with strict validation.

Environment variables:
    - GEMINI_API_KEY: Required for Gemini model access.

Author: Zohaib Khan
"""

from typing import Any
import os

from pydantic import BaseModel, ConfigDict
from dotenv import load_dotenv, find_dotenv
from openai import AsyncOpenAI
from agents import (
    RunContextWrapper,
    FunctionTool,
    Runner,
    Agent,
    OpenAIChatCompletionsModel,
    set_tracing_disabled,
)

# =========================
# Environment & Model Setup
# =========================

load_dotenv(find_dotenv())
set_tracing_disabled(True)  # Disable tracing output for now

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
    openai_client=external_client,
    model=GEMINI_MODEL_NAME,
)

print("✅ Model configured successfully\n")


# =========================
# Data Models
# =========================


class FunctionArgs(BaseModel):
    """
    Pydantic model for function arguments with strict validation.

    Uses ConfigDict(extra="forbid") to disallow unexpected fields.
    """

    username: str
    age: int

    model_config = ConfigDict(extra="forbid")  # disallow unexpected fields


# =========================
# Tool Functions
# =========================


def do_some_work(data: str) -> str:
    """
    Process user data with dummy logic.

    Args:
        data: String containing user information

    Returns:
        str: Processed result message
    """
    return f"✅ Work done for: {data}"


async def run_function(ctx: RunContextWrapper[Any], args: str) -> str:
    """
    Process user data using the provided arguments.

    Args:
        ctx: RunContextWrapper for managing execution context
        args: JSON string containing user data (username and age)

    Returns:
        str: Processed result or error message
    """
    try:
        parsed = FunctionArgs.model_validate_json(args)
        return do_some_work(data=f"{parsed.username} is {parsed.age} years old")
    except Exception as e:
        return f"❌ Error processing user data: {str(e)}"


# =========================
# Tool Configuration
# =========================

tool = FunctionTool(
    name="process_user",
    description="Processes extracted user data",
    params_json_schema=FunctionArgs.model_json_schema(),
    on_invoke_tool=run_function,
)


# =========================
# Agent Configuration
# =========================

agent = Agent(
    name="User Data Processor",
    instructions="You are a helpful assistant that can process user data using the `process_user` tool.",
    tools=[tool],
    model=model,
)


# =========================
# Execute Agent Run
# =========================

print("🔧 Tool Registration Details:")
print("=" * 40)

for tool in agent.tools:
    print("📦 Tool registered:")
    print(f"- Name       : {tool.name}")
    print(f"- Description: {tool.description}")
    print(f"- Schema     : {tool.params_json_schema}")
    print(f"- Function   : {tool.on_invoke_tool.__name__}\n")

print("🧪 Executing Agent with User Data...")
print("=" * 40)

try:
    # Execute agent with user data
    result = Runner.run_sync(
        agent,
        "Please use the process_user tool to process the user data: username='zohaib', age=19",
    )

    print("📤 Final Output:")
    print(result.final_output)

except Exception as e:
    print(f"❌ Error during agent execution: {e}")
    raise
