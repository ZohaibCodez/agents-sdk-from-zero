"""
01_hello_agent.py

Hello Agent Example
https://openai.github.io/openai-agents-python/agents/

Simple introduction to creating and running a basic agent with OpenAI Agents SDK.

Features demonstrated:
- Basic agent creation with name and instructions
- Simple agent execution with Runner.run()
- Basic response handling

Environment variables:
    - GEMINI_API_KEY: Required for Gemini model access.

Author: OpenAI Agents SDK Team
"""

import asyncio
import os
from dotenv import load_dotenv, find_dotenv

from openai import AsyncOpenAI

from agents import (
    Agent,
    Runner,
    OpenAIChatCompletionsModel,
    set_tracing_disabled,
    RunResult,
    RunContextWrapper,
)

# =========================
# Environment & Model Setup
# =========================

load_dotenv(find_dotenv())
set_tracing_disabled(True)

# Configuration constants
GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")

# Validate API key
if not GEMINI_API_KEY:
    print("❌ GEMINI_API_KEY environment variable is required but not found.")
    raise ValueError("GEMINI_API_KEY environment variable is required but not found.")

print(f"🚀 Initializing Gemini client with model: gemini-2.0-flash")

# Initialize external OpenAI client for Gemini
external_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Configure the chat completions model
model = OpenAIChatCompletionsModel(
    openai_client=external_client, model="gemini-2.0-flash"
)

print(f"✅ Model configured successfully\n")


# =========================
# Agent Definition
# =========================

hello_agent: Agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant.",
    model=model,
)


# =========================
# Main Execution
# =========================

async def main():
    """
    Demonstrate basic agent creation and execution.
    
    Creates a simple agent and runs it with a basic greeting.
    """
    try:
        print("\n" + "="*50)
        print("👋 HELLO AGENT DEMO")
        print("="*50)

        print("\n📥 Sending greeting to agent...")
        result = await Runner.run(hello_agent, "HI!")
        
        print(f"📤 Agent Response: {result.final_output}")
        print("\n✅ Hello agent demo completed successfully!")

    except Exception as e:
        print(f"❌ Error during execution: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
