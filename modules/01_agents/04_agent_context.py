"""
04_agent_context.py

Agent Context Example
https://openai.github.io/openai-agents-python/agents/

Demonstrates how to use context with agents to provide personalized responses.

Features demonstrated:
- Agent context usage with RunContextWrapper
- Dynamic instruction creation based on context
- Personalized agent responses

Environment variables:
    - GEMINI_API_KEY: Required for Gemini model access.

Author: OpenAI Agents SDK Team
"""

import asyncio
import os
from dotenv import load_dotenv, find_dotenv

from agents import (
    Agent,
    Runner,
    OpenAIChatCompletionsModel,
    set_tracing_disabled,
    RunResult,
    RunContextWrapper,
)
from openai import AsyncOpenAI

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
# Context Function
# =========================

def create_instruction(ctx: RunContextWrapper[str], agent: Agent[str]) -> str:
    """
    Create dynamic instructions based on user context.
    
    Args:
        ctx: Run context containing user information
        agent: The agent instance
        
    Returns:
        Personalized instruction string
    """
    try:
        user_name = ctx.context if ctx.context else "unknown user"
        print(f"👤 [CONTEXT]: {user_name}")
        return f"You are a helpful assistant that provides concise answers. You are talking to {user_name}"
    except Exception as e:
        print(f"❌ Error creating instruction: {e}")
        return "You are a helpful assistant that provides concise answers."


# =========================
# Agent Definition
# =========================

context_agent: Agent[str] = Agent[str](
    name="Context Agent", 
    instructions=create_instruction, 
    model=model
)


# =========================
# Main Execution
# =========================

async def main():
    """
    Demonstrate agent context usage with personalized responses.
    
    Creates an agent that uses context to provide personalized responses
    based on the user's name.
    """
    try:
        print("\n" + "="*50)
        print("👤 AGENT CONTEXT DEMO")
        print("="*50)

        # Test with a specific user name
        user_name = "Zohaib"
        print(f"\n📥 Sending greeting to agent for user: {user_name}")
        
        result: RunResult = await Runner.run(
            context_agent, 
            "Hi", 
            context=user_name
        )
        
        print(f"📤 Agent Response: {result.final_output}")
        print("\n✅ Agent context demo completed successfully!")

    except Exception as e:
        print(f"❌ Error during execution: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
