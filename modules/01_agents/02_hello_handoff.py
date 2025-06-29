"""
Multi-Agent Translation System with Handoff Capabilities

This module demonstrates agent handoff functionality where a triage agent
determines which specialized agent (French translator or Emojifier) should
handle the user's request based on the content.

Author: Zohaib Khan
"""

import os
import asyncio
from typing import Optional
from dotenv import load_dotenv, find_dotenv

from agents import (
    Agent,
    Runner,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    set_tracing_disabled,
)

# Load environment variables and disable tracing for cleaner output
load_dotenv(find_dotenv())
set_tracing_disabled(True)

# Configuration constants
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")
GEMINI_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_MODEL_NAME: str = "gemini-2.0-flash"

# Validate API key
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is required but not found.")

# Initialize external OpenAI client for Gemini
external_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url=GEMINI_BASE_URL,
)

# Configure the chat completions model
model = OpenAIChatCompletionsModel(
    openai_client=external_client, model=GEMINI_MODEL_NAME
)


async def run_translation_example() -> None:
    """
    Demonstrate the multi-agent translation system with handoff functionality.

    This function creates the agent hierarchy and runs a sample translation
    to showcase how the triage agent determines the appropriate specialized agent.
    """

    # Create specialized agents
    french_agent: Agent = Agent(
        name="French Translation Agent",
        instructions=(
            "Translate the sentence into poetic French, as if written by a romantic "
            "French poet. Add a touch of metaphor or emotion to make the translation "
            "more expressive and culturally authentic."
        ),
        handoff_description="Specialized agent for French Language Translation with poetic flair",
        model=model,
    )

    emojifier_agent: Agent = Agent(
        name="Emojifier Agent",
        instructions=(
            "Translate the meaning of the sentence into expressive emojis. "
            "Focus on capturing the emotional and semantic content through emoji combinations. "
            "Only use words if absolutely necessary for clarity."
        ),
        handoff_description="Specialized agent for converting text to expressive emoji sequences",
        model=model,
    )

    # Create triage agent with handoff capabilities
    triage_agent: Agent = Agent(
        name="Translation Triage Agent",
        instructions=(
            "You are a language translation coordinator. Your role is to analyze "
            "the user's request and determine which specialized agent should handle it. "
            "For French translation requests, hand off to the French agent. "
            "For emoji conversion requests, hand off to the Emojifier agent. "
            "Make clear, accurate translations when handling requests directly."
        ),
        handoffs=[emojifier_agent, french_agent],
        model=model,
    )

    # Sample text for translation demonstration
    sample_text = "I love programming because it feels like solving a puzzle."

    print(f"Input text: {sample_text}")
    print("-" * 50)

    try:
        # Run the translation through the agent system
        result = await Runner.run(triage_agent, sample_text)
        print(f"Translation result: {result.final_output}")
    except Exception as e:
        print(f"Error during translation: {e}")


if __name__ == "__main__":
    asyncio.run(run_translation_example())
