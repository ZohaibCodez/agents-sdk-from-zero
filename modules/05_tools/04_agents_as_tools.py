"""
Agents as Tools Demo - OpenAI Agents SDK

This module demonstrates how to use agents as tools within other agents.
It showcases:
- Creating specialized agents for different tasks
- Converting agents to tools using as_tool() method
- Orchestrating multiple agents through a main agent
- Content creation and editing workflows
- Error handling and graceful failure
- Multi-agent collaboration patterns

The demo implements a content editing system with specialized agents for poetry,
story writing, summarization, and grammar correction.

Environment variables:
    - OPENROUTER_API_KEY: Required for OpenRouter API access.

Author: Zohaib Khan
"""

import asyncio
import os
from agents.handoffs import Handoff, handoff
from dotenv import load_dotenv, find_dotenv
from openai import AsyncOpenAI
from openai.types.responses import ResponseTextDeltaEvent
from agents import (
    Agent,
    Runner,
    set_tracing_disabled
)
import random
from agents.extensions.models.litellm_model import LitellmModel

# =========================
# Environment & Model Setup
# =========================

load_dotenv(find_dotenv())
set_tracing_disabled(True)

# Configuration constants
MISTRAL_API_KEY: str | None = os.getenv("MISTRAL_API_KEY")
MODEL_NAME: str = "mistral/open-mixtral-8x7b"

# Validate API key
if not MISTRAL_API_KEY:
    print("❌ MISTRAL_API_KEY environment variable is required but not found.")
    raise ValueError("MISTRAL_API_KEY environment variable is required but not found.")

print(f"🚀 Initializing OpenRouter client with model: {MODEL_NAME}")

model = LitellmModel(model=MODEL_NAME, api_key=MISTRAL_API_KEY)


print(f"✅ Model configured successfully\n")


# =========================
# Specialized Agent Definitions
# =========================

poetry_agent = Agent(
    name="Poet",
    instructions="You're a skilled poet. Given a topic, write a beautiful short poem that reflects emotion and imagery.",
    model=model,
)

story_agent = Agent(
    name="StoryWriter",
    instructions="You're a creative short story writer. Write a brief but compelling story around the user's topic.",
    model=model,
)

summary_agent = Agent(
    name="Summarizer",
    instructions="You're a summarization expert. Given a long text, summarize it in 2-3 concise sentences.",
    model=model,
)

grammar_agent = Agent(
    name="GrammarFixer",
    instructions="You're a grammar expert. Correct grammatical errors and improve sentence clarity.",
    model=model,
)

# =========================
# Editor Orchestrator Agent
# =========================

editor_agent = Agent(
    name="Content Editor",
    instructions="""
    You are an editorial assistant. Depending on the user's request, delegate to one of the following specialists:
    - Poet: write short poems
    - StoryWriter: generate mini stories
    - Summarizer: summarize large texts
    - GrammarFixer: correct and improve text grammar

    You must always return the cleaned or generated content clearly.
    """,
    tools=[
        poetry_agent.as_tool(
            tool_name="write_poem",
            tool_description="Write a short poem on the given topic",
        ),
        story_agent.as_tool(
            tool_name="write_story",
            tool_description="Write a short story on the given topic",
        ),
        summary_agent.as_tool(
            tool_name="summarize_text",
            tool_description="Summarize a long passage into 2-3 lines",
        ),
        grammar_agent.as_tool(
            tool_name="fix_grammar",
            tool_description="Fix grammar mistakes and improve clarity",
        ),
    ],
    model=model,
)

# =========================
# Main Execution Function
# =========================

async def main():
    """
    Execute the multi-agent content editing demo.
    
    Demonstrates various content creation and editing tasks using
    specialized agents as tools within the main editor agent.
    """
    print("📝 Multi-Agent Content Editing Demo")
    print("=" * 50)
    
    try:
        # Test 1: Write a Poem
        print("\n🎭 Task 1: Write a Poem")
        print("-" * 30)
        result1 = await Runner.run(
            editor_agent, input="Write a poem about the beauty of a rainy day."
        )
        print(result1.final_output)
        print("\n" + "=" * 60 + "\n")

        # Test 2: Summarize Text
        print("📋 Task 2: Summarize Text")
        print("-" * 30)
        result2 = await Runner.run(
            editor_agent,
            input="Summarize this: 'Artificial intelligence is rapidly transforming industries. From healthcare to finance, it is helping automate tasks and generate insights that were previously impossible.'",
        )
        print(result2.final_output)
        print("\n" + "=" * 60 + "\n")

        # Test 3: Fix Grammar
        print("🔧 Task 3: Fix Grammar")
        print("-" * 30)
        result3 = await Runner.run(
            editor_agent,
            input="Fix the grammar: 'She no went to the market because it raining.'",
        )
        print(result3.final_output)
        print("\n" + "=" * 60 + "\n")

        # Test 4: Mixed Task
        print("🔄 Task 4: Mixed Task")
        print("-" * 30)
        result4 = await Runner.run(
            editor_agent,
            input="First fix grammar: 'The boy run fastly to school', then summarize this: 'School is a place where children not only learn but also develop social skills, discipline, and creativity.'",
        )
        print(result4.final_output)
        
        # Display agent information
        print("\n🤖 Agent Information:")
        print("-" * 30)
        print(f"Main Agent: {editor_agent.name}")
        print("Specialized Agents:")
        for tool in editor_agent.tools:
            print(f"  - {tool.name}: {tool.description}")
            
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 