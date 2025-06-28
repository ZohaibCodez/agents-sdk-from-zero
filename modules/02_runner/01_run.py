"""
Module: 01_run.py

Demonstrates basic agent execution using the OpenAI Agents SDK Runner.
Shows how to create a simple agent and run it with input, then examine
all the properties of the RunResult object.

Features demonstrated:
- Basic agent creation and execution
- RunResult object exploration
- Error handling for agent execution

Environment variables:
    - OPENROUTER_API_KEY: Required for OpenRouter API access.

Author: Zohaib Khan
"""

import asyncio
import os
from dotenv import load_dotenv, find_dotenv

from agents import (
    Agent,
    Runner,
    RunResult,
    set_tracing_disabled,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
)

# =========================
# Environment & Model Setup
# =========================

# Load environment variables from .env file
load_dotenv(find_dotenv())
set_tracing_disabled(True)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    print("OPENROUTER_API_KEY not found.")

external_client = AsyncOpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)

model = OpenAIChatCompletionsModel(
    openai_client=external_client, model="deepseek/deepseek-chat-v3-0324:free"
)

# =============================================================================
# Main Execution Function
# =============================================================================

async def run_basic_agent_demo():
    """
    Run a simple agent and explore all properties of the RunResult object.
    
    This function demonstrates:
    1. Creating a basic agent with specific instructions
    2. Running the agent with user input
    3. Examining all RunResult properties for educational purposes
    """
    print("--- Running 01_run.py --- ")

    # Create a simple agent with poetic capabilities
    assistant_agent = Agent(
        name="HaikuAssistant",
        instructions="You are a poetic assistant, skilled in writing haikus.",
        model=model,
    )

    # Define user input for the agent
    user_input = "Write a haiku about a diligent software engineer."

    # Run the agent asynchronously
    print(f"\n🤖 Assistant: Running agent with input: '{user_input}'")
    
    try:
        result: RunResult = await Runner.run(
            starting_agent=assistant_agent, input=user_input
        )

        # Explore all RunResult properties for educational purposes
        _explore_run_result_properties(result)

        print(f"\n--- Finished 01_run.py --- ")

    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print(f"--- Finished 01_run.py with error --- ")


def _explore_run_result_properties(result: RunResult):
    """
    Explore and display all properties of the RunResult object.
    
    This function is used for educational purposes to understand
    the structure and content of the RunResult object returned by
    the Runner.run() method.
    
    Args:
        result: The RunResult object to explore
    """
    print("\n" + "="*60)
    print("🔍 EXPLORING RUNRESULT PROPERTIES")
    print("="*60)
    
    # Basic result information
    print(f"\n📋 [RESULT OBJECT]\nType: {type(result)}\nValue: {result}")
    
    # Final output (most important)
    print(f"\n✅ [FINAL OUTPUT]\nType: {type(result.final_output)}\nValue: {result.final_output}")
    
    # Raw responses from the model
    print(f"\n🤖 [RAW RESPONSES]\nType: {type(result.raw_responses)}\nValue: {result.raw_responses}")
    
    # Guardrail results
    print(f"\n🛡️ [INPUT GUARDRAIL RESULTS]\nType: {type(result.input_guardrail_results)}\nValue: {result.input_guardrail_results}")
    
    print(f"\n🛡️ [OUTPUT GUARDRAIL RESULTS]\nType: {type(result.output_guardrail_results)}\nValue: {result.output_guardrail_results}")
    
    # Context and agent information
    print(f"\n📦 [CONTEXT WRAPPER]\nType: {type(result.context_wrapper)}\nValue: {result.context_wrapper}")
    
    print(f"\n🤖 [LAST AGENT]\nType: {type(result.last_agent)}\nValue: {result.last_agent}")
    
    print(f"\n🆔 [LAST RESPONSE ID]\nType: {type(result.last_response_id)}\nValue: {result.last_response_id}")
    
    # Items and input
    print(f"\n📝 [NEW ITEMS]\nType: {type(result.new_items)}\nValue: {result.new_items}")
    
    print(f"\n📥 [INPUT]\nType: {type(result.input)}\nValue: {result.input}")
    
    # Utility methods
    print(f"\n📋 [TO INPUT LIST]\nType: {type(result.to_input_list())}\nValue: {result.to_input_list()}")
    
    print(f"\n🔄 [FINAL OUTPUT AS STRING]\nType: {type(result.final_output_as(str))}\nValue: {result.final_output_as(str)}")
    
    print("\n" + "="*60)
    print("✅ PROPERTY EXPLORATION COMPLETE")
    print("="*60)


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    asyncio.run(run_basic_agent_demo())
